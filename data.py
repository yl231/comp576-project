import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
import json
import os
from typing import Tuple, Dict, List, Optional
import logging
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbedLLMDataProcessor:
    """
    Data processor for EmbedLLM datasets
    Handles loading, preprocessing, and preparation of data for training
    """
    
    def __init__(
        self,
        dataset_name: str,
        text_encoder: str,
        device: str = "cuda",
        model_type: str = "mlp",
    ):
        """
        Initialize the data processor
        
        Args:
            dataset_name: Name of the dataset (e.g., 'EmbedLLM')
            text_encoder: Text encoder model name
            device: Device to use for computations
        """
        self.dataset_name = dataset_name
        self.text_encoder = text_encoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder_name = text_encoder.replace("/", "_")
        self.model_type = model_type
        self._pairwise_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.llm_embeddings: Optional[torch.Tensor] = None

        # Define paths
        self.dataset_dir = f"./datasets/{dataset_name}"
        self.encoded_dir = f"./datasets/{dataset_name}/{self.encoder_name}"
        self.llm_embeddings_path = os.path.join(
            self.encoded_dir,
            "llm_embedding.pt"
        )
        self.llm_descriptions_json = os.path.join(self.dataset_dir, "llm_descriptions.json")
        self.llm_descriptions_csv = os.path.join(self.dataset_dir, "llm_descriptions.csv")
        
        # Initialize text encoder
        self.text_encoder_model = None
        
        logger.info(f"Initialized EmbedLLMDataProcessor for dataset: {dataset_name}")
        logger.info(f"Text encoder: {text_encoder}")
        logger.info(f"Using device: {self.device}")

        self._load_or_create_llm_embeddings()
    
    def set_llm_embeddings(self, llm_embeddings: torch.Tensor) -> None:
        """
        Set precomputed LLM embeddings used for MIRT pairwise data generation.
        """
        if llm_embeddings.dim() != 2:
            raise ValueError(f"llm_embeddings must be 2D (n_models, embedding_dim); got shape {llm_embeddings.shape}")
        self.llm_embeddings = llm_embeddings.float().contiguous()
        self._pairwise_cache.clear()

        # Persist embeddings for future runs
        os.makedirs(os.path.dirname(self.llm_embeddings_path), exist_ok=True)
        torch.save(self.llm_embeddings, self.llm_embeddings_path)
        logger.info(f"Saved LLM embeddings to {self.llm_embeddings_path}")

    def _load_or_create_llm_embeddings(self) -> None:
        if os.path.exists(self.llm_embeddings_path):
            self.llm_embeddings = torch.load(self.llm_embeddings_path, map_location="cpu").float().contiguous()
            logger.info(f"Loaded LLM embeddings from {self.llm_embeddings_path} with shape {self.llm_embeddings.shape}")
            return

        model_ids, descriptions = self._load_llm_descriptions()
        if not descriptions:
            raise FileNotFoundError(
                f"LLM embeddings are required for MIRT but neither {self.llm_embeddings_path} "
                "nor a description file was found."
            )

        encoder = self._get_text_encoder()
        logger.info(
            f"Generating LLM embeddings from descriptions using encoder '{self.text_encoder}'. "
            f"Saving to {self.llm_embeddings_path}"
        )
        sentence_embeddings = encoder.encode(
            descriptions,
            show_progress_bar=True,
            convert_to_tensor=True,
            batch_size=min(2048, len(descriptions))
        )
        sentence_embeddings = sentence_embeddings.to("cpu", dtype=torch.float32)

        num_models = max(model_ids) + 1
        embedding_dim = sentence_embeddings.shape[1]
        llm_embeddings = torch.zeros((num_models, embedding_dim), dtype=torch.float32)

        id_set = set(model_ids)
        for idx, model_id in enumerate(model_ids):
            llm_embeddings[model_id] = sentence_embeddings[idx]

        missing_ids = [model_id for model_id in range(num_models) if model_id not in id_set]
        if missing_ids:
            raise ValueError(
                "LLM descriptions are missing entries for the following model IDs: "
                f"{missing_ids}. Please ensure the descriptions file lists every model."
            )

        os.makedirs(os.path.dirname(self.llm_embeddings_path), exist_ok=True)
        torch.save(llm_embeddings, self.llm_embeddings_path)
        logger.info(f"Saved generated LLM embeddings to {self.llm_embeddings_path}")

        self.llm_embeddings = llm_embeddings.contiguous()
        self._pairwise_cache.clear()

    def _load_llm_descriptions(self) -> Tuple[List[int], List[str]]:
        entries: List[Dict[str, str]] = []
        if os.path.exists(self.llm_descriptions_json):
            with open(self.llm_descriptions_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "llms" in data:
                    entries = data["llms"]
                else:
                    entries = list(data.values())
            elif isinstance(data, list):
                entries = data
            else:
                raise ValueError(f"Unsupported JSON format in {self.llm_descriptions_json}")
        elif os.path.exists(self.llm_descriptions_csv):
            df = pd.read_csv(self.llm_descriptions_csv)
            if not {"model_id", "description"}.issubset(df.columns):
                raise ValueError(
                    f"{self.llm_descriptions_csv} must contain 'model_id' and 'description' columns."
                )
            entries = df.to_dict("records")
        else:
            return [], []

        model_ids: List[int] = []
        descriptions: List[str] = []
        for entry in entries:
            if "model_id" not in entry or "description" not in entry:
                raise ValueError("Each LLM description entry must include 'model_id' and 'description'.")
            model_ids.append(int(entry["model_id"]))
            descriptions.append(str(entry["description"]))

        if not model_ids:
            return [], []

        sorted_pairs = sorted(zip(model_ids, descriptions), key=lambda x: x[0])
        model_ids_sorted, descriptions_sorted = zip(*sorted_pairs)
        return list(model_ids_sorted), list(descriptions_sorted)
    
    def load_raw_data(self, data_type: str = "train") -> pd.DataFrame:
        """
        Load raw CSV data from the dataset directory
        
        Args:
            data_type: 'train' or 'test'
            
        Returns:
            DataFrame with the raw data
        """
        csv_path = f"{self.dataset_dir}/{data_type}.csv"
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        logger.info(f"Loading {data_type} data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        return df
    
    def _get_text_encoder(self):
        """Initialize and return the text encoder model"""
        if self.text_encoder_model is None:
            logger.info(f"Loading text encoder: {self.text_encoder}")
            self.text_encoder_model = SentenceTransformer(self.text_encoder)
            self.text_encoder_model = self.text_encoder_model.to(self.device)
        return self.text_encoder_model
    
    def create_encoded_data_from_raw(self, data_type: str = "train") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Create encoded data from raw CSV files following the process_and_save_data.py approach
        
        Args:
            data_type: 'train', 'val', or 'test'
            
        Returns:
            tuple: (labels_tensor, embeddings_tensor, text_list)
        """
        logger.info(f"Creating encoded data from raw CSV for {data_type}")
        
        # Load raw data
        df = self.load_raw_data(data_type)
        
        # Optimize: Use more efficient aggregation (group by model_id, prompt_id, prompt and take max label)
        df = df.groupby(['model_id', 'prompt_id', 'prompt'])['label'].max().reset_index()
        
        # Get unique questions once and create mapping
        unique_questions = df[['prompt_id', 'prompt']].drop_duplicates().reset_index(drop=True)
        
        # Initialize embedder
        logger.info(f"Initializing sentence transformer: {self.text_encoder}")
        encoder = self._get_text_encoder()
        
        # Batch process embeddings for all unique questions
        batch_size = 10000  # Adjust based on GPU memory
        logger.info(f"Processing {len(unique_questions)} unique questions in batches of {batch_size}...")
        all_embeddings = []
        
        for i in range(0, len(unique_questions), batch_size):
            batch_prompts = unique_questions['prompt'].iloc[i:i+batch_size].tolist()
            batch_embeddings = encoder.encode(batch_prompts, show_progress_bar=True, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Stack all embeddings
        embeddings_array = np.vstack(all_embeddings)
        
        # Create labels matrix (model_num, query_num) where values are actual labels
        logger.info("Creating labels matrix...")
        labels_matrix = df.pivot(index='model_id', columns='prompt_id', values='label').fillna(0).astype(int)
        labels_array = labels_matrix.values
        
        # Create embeddings tensor corresponding to query_num dimension
        # Map prompt_ids to their embedding indices
        prompt_id_to_embedding_idx = {prompt_id: idx for idx, prompt_id in enumerate(unique_questions['prompt_id'])}
        
        # Get embeddings in the same order as the labels matrix columns
        embedding_indices = [prompt_id_to_embedding_idx[prompt_id] for prompt_id in labels_matrix.columns]
        ordered_embeddings = embeddings_array[embedding_indices]
        
        # Create text list in the same order as embeddings
        prompt_id_to_text = {prompt_id: text for prompt_id, text in zip(unique_questions['prompt_id'], unique_questions['prompt'])}
        text_list = [prompt_id_to_text[prompt_id] for prompt_id in labels_matrix.columns]
        
        # Convert to tensors
        labels_tensor = torch.tensor(labels_array, dtype=torch.float32)
        embeddings_tensor = torch.tensor(ordered_embeddings, dtype=torch.float32)
        
        logger.info(f"Created encoded data:")
        logger.info(f"  Labels shape: {labels_tensor.shape}")
        logger.info(f"  Embeddings shape: {embeddings_tensor.shape}")
        logger.info(f"  Text list length: {len(text_list)}")
        logger.info(f"Labels matrix shape: (models={labels_tensor.shape[0]}, queries={labels_tensor.shape[1]})")
        logger.info(f"Embeddings tensor shape: (queries={embeddings_tensor.shape[0]}, embedding_dim={embeddings_tensor.shape[1]})")
        
        return labels_tensor, embeddings_tensor, text_list
    
    def load_encoded_data(self, data_type: str = "train") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Load pre-encoded data from the text encoder specific directory.
        If encoded data doesn't exist, create it from raw CSV files.
        
        Args:
            data_type: 'train', 'val', or 'test'
            
        Returns:
            tuple: (labels_tensor, embeddings_tensor, text_list)
        """
        encoded_path = f"{self.encoded_dir}/{data_type}"
        
        # Check if encoded data exists
        labels_path = f"{encoded_path}/labels_tensor.pt"
        embeddings_path = f"{encoded_path}/embeddings_tensor.pt"
        text_list_path = f"{encoded_path}/text_list.pkl"
        
        if not all(os.path.exists(p) for p in [labels_path, embeddings_path, text_list_path]):
            logger.info(f"Encoded data not found for {data_type}, creating from raw CSV...")
            
            # Create encoded data from raw CSV
            labels_tensor, embeddings_tensor, text_list = self.create_encoded_data_from_raw(data_type)
            
            # Save the encoded data for future use
            self.save_encoded_data(labels_tensor, embeddings_tensor, text_list, data_type)
            
            return labels_tensor, embeddings_tensor, text_list
        
        # Load existing encoded data
        labels_tensor = torch.load(labels_path, map_location=torch.device("cpu"))
        embeddings_tensor = torch.load(embeddings_path, map_location=torch.device("cpu"))
        
        with open(text_list_path, "rb") as f:
            text_list = pickle.load(f)
        
        logger.info(f"Loaded {data_type} encoded data:")
        logger.info(f"  Labels shape: {labels_tensor.shape}")
        logger.info(f"  Embeddings shape: {embeddings_tensor.shape}")
        logger.info(f"  Text list length: {len(text_list)}")
        
        return labels_tensor, embeddings_tensor, text_list
    
    def prepare_training_data(self, 
                             labels_tensor: torch.Tensor, 
                             embeddings_tensor: torch.Tensor, 
                             text_list: List[str],
                             batch_size: int = 32,
                             shuffle: bool = True) -> Dict[str, DataLoader]:
        """
        Prepare data for training with multi-output MLP
        
        Args:
            labels_tensor: (n_models, n_queries) tensor of labels
            embeddings_tensor: (n_queries, embedding_dim) tensor of embeddings
            text_list: List of text prompts
            batch_size: Batch size for data loaders
            shuffle: Whether to shuffle training data
            
        Returns:
            dict: Dictionary containing training data loader
        """
        logger.info("Preparing data for training...")
        if self.model_type == "mirt":
            return self._prepare_pairwise_training_data(
                labels_tensor=labels_tensor,
                embeddings_tensor=embeddings_tensor,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        
        n_models, n_queries = labels_tensor.shape
        embedding_dim = embeddings_tensor.shape[1]
        
        logger.info(f"Data shapes: {n_models} models, {n_queries} queries, {embedding_dim} embedding dim")
        
        # For multi-output MLP: each query gets all model predictions
        # X: (n_queries, embedding_dim) - one embedding per query
        # y: (n_queries, n_models) - all model predictions for each query
        X = embeddings_tensor  # (n_queries, embedding_dim)
        y = labels_tensor.T    # (n_queries, n_models) - transpose to get queries x models
        
        logger.info(f"Multi-output data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Each sample: embedding_dim={embedding_dim}, n_models={n_models}")
        
        # Create data loader
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        logger.info(f"Created training data loader with batch size: {batch_size}")
        logger.info(f"Each batch will have shape: X=(batch_size, {embedding_dim}), y=(batch_size, {n_models})")
        
        return {
            'train_loader': train_loader,
            'train_dataset': train_dataset,
            'n_models': n_models,
            'embedding_dim': embedding_dim
        }
    
    def prepare_test_data(self, 
                         labels_tensor: torch.Tensor, 
                         embeddings_tensor: torch.Tensor, 
                         text_list: List[str],
                         batch_size: int = 32) -> DataLoader:
        """
        Prepare test data for evaluation with multi-output MLP
        
        Args:
            labels_tensor: (n_models, n_queries) tensor of labels
            embeddings_tensor: (n_queries, embedding_dim) tensor of embeddings
            text_list: List of text prompts
            batch_size: Batch size for data loader
            
        Returns:
            DataLoader for test data
        """
        logger.info("Preparing test data...")
        if self.model_type == "mirt":
            return self._prepare_pairwise_test_loader(
                labels_tensor=labels_tensor,
                embeddings_tensor=embeddings_tensor,
                batch_size=batch_size,
            )
        
        n_models, n_queries = labels_tensor.shape
        embedding_dim = embeddings_tensor.shape[1]
        
        logger.info(f"Test data shapes: {n_models} models, {n_queries} queries, {embedding_dim} embedding dim")
        
        # For multi-output MLP: each query gets all model predictions
        # X: (n_queries, embedding_dim) - one embedding per query
        # y: (n_queries, n_models) - all model predictions for each query
        X = embeddings_tensor  # (n_queries, embedding_dim)
        y = labels_tensor.T    # (n_queries, n_models) - transpose to get queries x models
        
        logger.info(f"Multi-output test data shape: X={X.shape}, y={y.shape}")
        
        # Create test dataset and loader
        test_dataset = TensorDataset(X, y)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created test data loader with batch size: {batch_size}")
        logger.info(f"Each batch will have shape: X=(batch_size, {embedding_dim}), y=(batch_size, {n_models})")
        
        return test_loader

    def _get_pairwise_tensors(
        self,
        labels_tensor: torch.Tensor,
        embeddings_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert matrix-form labels and embeddings into pairwise tensors.
        """
        cache_key = (labels_tensor.data_ptr(), embeddings_tensor.data_ptr())
        if cache_key in self._pairwise_cache:
            return self._pairwise_cache[cache_key]

        labels_tensor = labels_tensor.contiguous()
        embeddings_tensor = embeddings_tensor.contiguous()

        n_models, n_queries = labels_tensor.shape
        embedding_dim = embeddings_tensor.shape[1]

        if self.llm_embeddings is None:
            raise ValueError(
                "LLM embeddings are not set. Call `set_llm_embeddings` with a "
                f"(n_models={n_models}, embedding_dim={embedding_dim}) tensor before using the MIRT pipeline."
            )
        if self.llm_embeddings.shape != (n_models, embedding_dim):
            raise ValueError(
                "LLM embeddings shape mismatch. Expected "
                f"({n_models}, {embedding_dim}), got {self.llm_embeddings.shape}."
            )

        llm_features = self.llm_embeddings.repeat_interleave(n_queries, dim=0)
        query_embeddings = embeddings_tensor.repeat(n_models, 1)
        responses = labels_tensor.reshape(-1)

        pairwise = (llm_features, query_embeddings, responses)
        self._pairwise_cache[cache_key] = pairwise
        return pairwise

    def _prepare_pairwise_training_data(
        self,
        labels_tensor: torch.Tensor,
        embeddings_tensor: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Dict[str, DataLoader]:
        """
        Prepare pairwise (llm, query) data suitable for MIRT models.
        """
        logger.info("Preparing pairwise data for MIRT...")

        llm_features, query_embeddings, responses = self._get_pairwise_tensors(labels_tensor, embeddings_tensor)

        n_models = labels_tensor.shape[0]
        embedding_dim = embeddings_tensor.shape[1]

        logger.info(f"Pairwise data shapes: {n_models} models, embedding_dim={embedding_dim}, total_pairs={len(llm_features)}")

        dataset = TensorDataset(llm_features, query_embeddings, responses)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return {
            'train_loader': loader,
            'train_dataset': dataset,
            'n_models': n_models,
            'embedding_dim': embedding_dim
        }

    def _prepare_pairwise_test_loader(
        self,
        labels_tensor: torch.Tensor,
        embeddings_tensor: torch.Tensor,
        batch_size: int = 32,
    ) -> DataLoader:
        """
        Prepare pairwise data loader for validation or testing.
        """
        llm_features, query_embeddings, responses = self._get_pairwise_tensors(labels_tensor, embeddings_tensor)

        dataset = TensorDataset(llm_features, query_embeddings, responses)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def save_encoded_data(self, 
                         labels_tensor: torch.Tensor, 
                         embeddings_tensor: torch.Tensor, 
                         text_list: List[str],
                         data_type: str = "train") -> None:
        """
        Save encoded data to the text encoder specific directory
        
        Args:
            labels_tensor: (n_models, n_queries) tensor of labels
            embeddings_tensor: (n_queries, embedding_dim) tensor of embeddings
            text_list: List of text prompts
            data_type: 'train' or 'test'
        """
        # Create directory if it doesn't exist
        save_dir = f"{self.encoded_dir}/{data_type}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save tensors
        torch.save(labels_tensor, f"{save_dir}/labels_tensor.pt")
        torch.save(embeddings_tensor, f"{save_dir}/embeddings_tensor.pt")
        
        # Save text list
        with open(f"{save_dir}/text_list.pkl", "wb") as f:
            pickle.dump(text_list, f)
        
        # Save encoder info
        encoder_info = {
            "text_encoder": self.text_encoder,
            "encoder_name": self.encoder_name,
            "data_type": data_type,
            "n_models": labels_tensor.shape[0],
            "n_queries": labels_tensor.shape[1],
            "embedding_dim": embeddings_tensor.shape[1]
        }
        
        with open(f"{save_dir}/encoder_info.json", "w") as f:
            json.dump(encoder_info, f, indent=2)
        
        logger.info(f"Saved encoded {data_type} data to {save_dir}")
        logger.info(f"Encoder info: {encoder_info}")


class RouterBenchDataProcessor:
    """
    Data processor for RouterBench datasets
    Handles loading, preprocessing, and preparation of data for training
    Compatible with comp576-routers main.py interface
    """
    
    def __init__(
        self,
        dataset_name: str,
        text_encoder: str,
        device: str = "cuda",
        model_type: str = "mlp",
    ):
        """
        Initialize the data processor
        
        Args:
            dataset_name: Name of the dataset (e.g., 'RouterBench')
            text_encoder: Text encoder model name
            device: Device to use for computations
            model_type: Type of model ('mlp', 'mirt', 'knn')
        """
        self.dataset_name = dataset_name
        self.text_encoder = text_encoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder_name = text_encoder.replace("/", "_")
        self.model_type = model_type
        self._pairwise_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.llm_embeddings: Optional[torch.Tensor] = None

        # Define paths
        self.dataset_dir = f"./datasets/{dataset_name}"
        self.encoded_dir = f"./datasets/{dataset_name}/{self.encoder_name}"
        self.llm_embeddings_path = os.path.join(
            self.encoded_dir,
            "llm_embedding.pt"
        )
        self.llm_descriptions_json = os.path.join(self.dataset_dir, "llm_descriptions.json")
        self.llm_descriptions_csv = os.path.join(self.dataset_dir, "llm_descriptions.csv")
        
        # Initialize text encoder
        self.text_encoder_model = None
        
        logger.info(f"Initialized RouterBenchDataProcessor for dataset: {dataset_name}")
        logger.info(f"Text encoder: {text_encoder}")
        logger.info(f"Using device: {self.device}")

        self._load_or_create_llm_embeddings()
    
    def set_llm_embeddings(self, llm_embeddings: torch.Tensor) -> None:
        """
        Set precomputed LLM embeddings used for MIRT pairwise data generation.
        """
        if llm_embeddings.dim() != 2:
            raise ValueError(f"llm_embeddings must be 2D (n_models, embedding_dim); got shape {llm_embeddings.shape}")
        self.llm_embeddings = llm_embeddings.float().contiguous()
        self._pairwise_cache.clear()

        # Persist embeddings for future runs
        os.makedirs(os.path.dirname(self.llm_embeddings_path), exist_ok=True)
        torch.save(self.llm_embeddings, self.llm_embeddings_path)
        logger.info(f"Saved LLM embeddings to {self.llm_embeddings_path}")

    def _load_or_create_llm_embeddings(self) -> None:
        if os.path.exists(self.llm_embeddings_path):
            self.llm_embeddings = torch.load(self.llm_embeddings_path, map_location="cpu").float().contiguous()
            logger.info(f"Loaded LLM embeddings from {self.llm_embeddings_path} with shape {self.llm_embeddings.shape}")
            return

        model_ids, descriptions = self._load_llm_descriptions()
        if not descriptions:
            self._handle_missing_llm_descriptions()
            return

        encoder = self._get_text_encoder()
        logger.info(
            f"Generating LLM embeddings from descriptions using encoder '{self.text_encoder}'. "
            f"Saving to {self.llm_embeddings_path}"
        )
        sentence_embeddings = encoder.encode(
            descriptions,
            show_progress_bar=True,
            convert_to_tensor=True,
            batch_size=min(2048, len(descriptions))
        )
        sentence_embeddings = sentence_embeddings.to("cpu", dtype=torch.float32)

        num_models = max(model_ids) + 1
        embedding_dim = sentence_embeddings.shape[1]
        llm_embeddings = torch.zeros((num_models, embedding_dim), dtype=torch.float32)

        id_set = set(model_ids)
        for idx, model_id in enumerate(model_ids):
            llm_embeddings[model_id] = sentence_embeddings[idx]

        missing_ids = [model_id for model_id in range(num_models) if model_id not in id_set]
        if missing_ids:
            raise ValueError(
                "LLM descriptions are missing entries for the following model IDs: "
                f"{missing_ids}. Please ensure the descriptions file lists every model."
            )

        os.makedirs(os.path.dirname(self.llm_embeddings_path), exist_ok=True)
        torch.save(llm_embeddings, self.llm_embeddings_path)
        logger.info(f"Saved generated LLM embeddings to {self.llm_embeddings_path}")

        self.llm_embeddings = llm_embeddings.contiguous()
        self._pairwise_cache.clear()

    def _handle_missing_llm_descriptions(self) -> None:
        """
        RouterBench-specific: Allow missing LLM descriptions (not required for all model types).
        """
        logger.debug(
            f"LLM embeddings not found at {self.llm_embeddings_path} and no description file available. "
            "Will be required for MIRT model type."
        )
        # Don't raise error - embeddings are only needed for MIRT

    def _load_llm_descriptions(self) -> Tuple[List[int], List[str]]:
        entries: List[Dict[str, str]] = []
        if os.path.exists(self.llm_descriptions_json):
            with open(self.llm_descriptions_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "llms" in data:
                    entries = data["llms"]
                else:
                    entries = list(data.values())
            elif isinstance(data, list):
                entries = data
            else:
                raise ValueError(f"Unsupported JSON format in {self.llm_descriptions_json}")
        elif os.path.exists(self.llm_descriptions_csv):
            df = pd.read_csv(self.llm_descriptions_csv)
            if not {"model_id", "description"}.issubset(df.columns):
                raise ValueError(
                    f"{self.llm_descriptions_csv} must contain 'model_id' and 'description' columns."
                )
            entries = df.to_dict("records")
        else:
            return [], []

        model_ids: List[int] = []
        descriptions: List[str] = []
        for entry in entries:
            if "model_id" not in entry or "description" not in entry:
                raise ValueError("Each LLM description entry must include 'model_id' and 'description'.")
            model_ids.append(int(entry["model_id"]))
            descriptions.append(str(entry["description"]))

        if not model_ids:
            return [], []

        sorted_pairs = sorted(zip(model_ids, descriptions), key=lambda x: x[0])
        model_ids_sorted, descriptions_sorted = zip(*sorted_pairs)
        return list(model_ids_sorted), list(descriptions_sorted)

    def _download_and_process_dataset(self) -> None:
        """
        Download RouterBench dataset from HuggingFace Hub and convert to CSV format.
        Downloads the 5-shot dataset, processes it, and creates train/val/test splits.
        """
        from huggingface_hub import hf_hub_download
        from sklearn.model_selection import train_test_split

        repo_id = "withmartian/routerbench"
        filename = "routerbench_5shot.pkl"

        # Ensure dataset directory exists
        os.makedirs(self.dataset_dir, exist_ok=True)

        logger.info(f"Downloading RouterBench dataset from {repo_id}...")

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=self.dataset_dir,
            )
            logger.info(f"✓ Downloaded to: {local_path}")
        except Exception as e:
            logger.error(f"Failed to download RouterBench dataset: {e}")
            raise

        # Load the dataset
        logger.info("Loading and processing dataset...")
        df = pd.read_pickle(local_path)
        logger.info(f"✓ Dataset loaded! Shape: {df.shape}, Columns: {len(df.columns)}")

        # RouterBench structure:
        # - sample_id contains split info (e.g., "Chinese_character_riddles.dev.0")
        # - Base model columns (without |) contain labels (0.0 or 1.0)
        # - Cost columns are named "model_name|total_cost"
        # - Response columns are named "model_name|model_response"

        # Exclude metadata columns
        metadata_cols = ["sample_id", "prompt", "eval_name", "oracle_model_to_route_to"]

        # Identify base model columns (label columns - without | separator)
        all_cols = df.columns.tolist()
        base_model_cols = [
            col for col in all_cols if col not in metadata_cols and "|" not in col
        ]

        # Extract model names from base model columns
        model_names = sorted(base_model_cols)
        n_models = len(model_names)
        logger.info(
            f"Found {n_models} models: {model_names[:5]}{'...' if n_models > 5 else ''}"
        )

        # Create model_id mapping
        model_name_to_id = {name: idx for idx, name in enumerate(model_names)}

        # Create mapping from model name to cost column name
        model_to_cost_col: dict[str, Optional[str]] = {}
        for model_name in model_names:
            # Find corresponding cost column
            cost_column_name = f"{model_name}|total_cost"
            if cost_column_name in df.columns:
                model_to_cost_col[model_name] = cost_column_name
            else:
                logger.warning(f"No cost column found for model {model_name}")
                model_to_cost_col[model_name] = None

        # Process the data to long format: each row is (prompt_id, model_id, label, cost, ...)
        rows = []
        for idx, row in df.iterrows():
            sample_id = str(
                row["sample_id"]
            )  # Keep as string since it contains split info
            prompt = row["prompt"]
            eval_name = row.get("eval_name", "")

            # Use sample_id as prompt_id (it's already unique and contains split info)
            prompt_id = sample_id

            # For each model, extract label and cost
            for model_name in model_names:
                model_id = model_name_to_id[model_name]

                # Extract label from base model column (can be any value between 0.0 and 1.0)
                label_value = row[model_name]
                if isinstance(label_value, (int, float)):
                    label = float(label_value)
                elif isinstance(label_value, bool):
                    label = 1.0 if label_value else 0.0
                else:
                    logger.warning(
                        f"Unexpected label type for {model_name} in row {idx}: {type(label_value)}"
                    )
                    label = 0.0

                # Extract cost from cost column
                cost_col: Optional[str] = model_to_cost_col.get(model_name)
                if cost_col is not None and cost_col in row:
                    cost = float(row[cost_col])
                else:
                    cost = 0.0

                rows.append(
                    {
                        "prompt_id": prompt_id,  # Keep as string
                        "prompt": str(prompt),
                        "eval_name": str(eval_name),
                        "category_id": 0,
                        "category": str(eval_name),
                        "model_name": model_name,
                        "model_id": model_id,
                        "label": int(label),
                        "cost": float(cost),
                    }
                )

        # Create DataFrame in long format
        processed_df = pd.DataFrame(rows)
        logger.info(f"✓ Processed to long format: {len(processed_df)} rows")

        # Create train/val/test splits: 60-10-30 using random sampling
        # First, get unique prompt_ids for splitting (to avoid data leakage)
        unique_prompt_ids = processed_df["prompt_id"].unique()
        n_unique_prompts = len(unique_prompt_ids)

        logger.info(f"Total unique prompts: {n_unique_prompts}")

        # First split: 60% train, 40% temp (which will become 10% val + 30% test)
        train_prompt_ids, temp_prompt_ids = train_test_split(
            unique_prompt_ids, test_size=0.4, random_state=42
        )

        # Second split: 40% temp -> 10% val (25% of temp) + 30% test (75% of temp)
        val_prompt_ids, test_prompt_ids = train_test_split(
            temp_prompt_ids, test_size=0.75, random_state=42
        )

        # Filter DataFrames by prompt_id sets
        train_df = processed_df[processed_df["prompt_id"].isin(train_prompt_ids)].copy()
        val_df = processed_df[processed_df["prompt_id"].isin(val_prompt_ids)].copy()
        test_df = processed_df[processed_df["prompt_id"].isin(test_prompt_ids)].copy()

        logger.info("✓ Splits created:")
        logger.info(
            f"  Train: {len(train_df)} samples ({len(train_df) / len(processed_df) * 100:.1f}%)"
        )
        logger.info(
            f"  Val: {len(val_df)} samples ({len(val_df) / len(processed_df) * 100:.1f}%)"
        )
        logger.info(
            f"  Test: {len(test_df)} samples ({len(test_df) / len(processed_df) * 100:.1f}%)"
        )

        # Save CSV files
        train_df.to_csv(f"{self.dataset_dir}/train.csv", index=False)
        val_df.to_csv(f"{self.dataset_dir}/val.csv", index=False)
        test_df.to_csv(f"{self.dataset_dir}/test.csv", index=False)

        logger.info(f"✓ Saved CSV files to {self.dataset_dir}/")

        # Save model_order.csv for consistency with EmbedLLM format
        model_order_df = pd.DataFrame(
            {
                "model_id": list(model_name_to_id.values()),
                "model_name": list(model_name_to_id.keys()),
            }
        )
        model_order_df.to_csv(f"{self.dataset_dir}/model_order.csv", index=False)
        logger.info("✓ Saved model_order.csv")

    def load_raw_data(self, data_type: str = "train") -> pd.DataFrame:
        """
        Load raw CSV data from the dataset directory.
        Automatically downloads and processes the dataset from HuggingFace Hub if files are missing.

        Args:
            data_type: 'train', 'val', or 'test'

        Returns:
            DataFrame with the raw data
        """
        csv_path = f"{self.dataset_dir}/{data_type}.csv"

        if not os.path.exists(csv_path):
            logger.warning(f"Data file not found: {csv_path}")
            logger.info("Attempting to download and process RouterBench dataset...")
            self._download_and_process_dataset()

            # Check again after download
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Data file not found after download attempt: {csv_path}"
                )

        logger.debug(f"Loading {data_type} data from {csv_path}")
        df = pd.read_csv(csv_path)

        # RouterBench CSV has columns: prompt_id, prompt, eval_name, category_id, category, model_name, model_id, label, cost
        # Verify required columns exist
        required_columns = ["prompt_id", "model_id", "label", "prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {csv_path}: {missing_columns}"
            )

        logger.debug(f"Loaded {data_type} data: {len(df)} samples")
        return df

    def load_cost_matrix(
        self, data_type: str = "test", labels_matrix: Optional[pd.DataFrame] = None
    ) -> Optional[torch.Tensor]:
        """
        Load cost matrix for RouterBench dataset.
        Returns a (n_models, n_queries) tensor with cost values.
        Returns None if dataset is not RouterBench or cost information is not available.

        Args:
            data_type: 'train', 'val', or 'test'
            labels_matrix: Optional DataFrame with model_id as index and prompt_id as columns.
                          If provided, uses its ordering. If None, loads raw data to determine ordering.

        Returns:
            Cost tensor of shape (n_models, n_queries) or None
        """

        csv_path = f"{self.dataset_dir}/{data_type}.csv"
        if not os.path.exists(csv_path):
            logger.warning(f"Cost matrix: CSV file not found: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            if "cost" not in df.columns:
                logger.warning(f"Cost matrix: 'cost' column not found in {csv_path}")
                return None

            # Group by model_id, prompt_id and take mean cost (in case of duplicates)
            df_cost = df.groupby(["model_id", "prompt_id"])["cost"].mean().reset_index()

            # Create cost matrix (model_id, prompt_id) -> cost
            cost_matrix_df = (
                df_cost.pivot(index="model_id", columns="prompt_id", values="cost")
                .fillna(0.0)
                .astype(float)
            )

            # If labels_matrix is provided, reindex to match its ordering
            if labels_matrix is not None:
                cost_matrix_df = cost_matrix_df.reindex(
                    index=labels_matrix.index,
                    columns=labels_matrix.columns,
                    fill_value=0.0,
                )

            cost_array = cost_matrix_df.values

            # Convert to tensor
            cost_tensor = torch.tensor(cost_array, dtype=torch.float32)

            logger.debug(
                f"Loaded cost matrix: {cost_tensor.shape}, "
                f"total cost range: [{cost_tensor.min():.8f}, {cost_tensor.max():.8f}]"
            )

            return cost_tensor
        except (pd.errors.ParserError, KeyError, ValueError, RuntimeError) as e:
            logger.warning(f"Failed to load cost matrix: {e}")
            return None

    def _get_text_encoder(self):
        """Initialize and return the text encoder model"""
        if self.text_encoder_model is None:
            logger.info(f"Loading text encoder: {self.text_encoder}")
            self.text_encoder_model = SentenceTransformer(self.text_encoder)
            self.text_encoder_model = self.text_encoder_model.to(self.device)
        return self.text_encoder_model
    
    def create_encoded_data_from_raw(self, data_type: str = "train") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Create encoded data from raw CSV files following the process_and_save_data.py approach
        
        Args:
            data_type: 'train', 'val', or 'test'
            
        Returns:
            tuple: (labels_tensor, embeddings_tensor, text_list)
        """
        logger.info(f"Creating encoded data from raw CSV for {data_type}")
        
        # Load raw data
        df = self.load_raw_data(data_type)
        
        # Optimize: Use more efficient aggregation (group by model_id, prompt_id, prompt and take max label)
        df = df.groupby(['model_id', 'prompt_id', 'prompt'])['label'].max().reset_index()
        
        # Get unique questions once and create mapping
        unique_questions = df[['prompt_id', 'prompt']].drop_duplicates().reset_index(drop=True)
        
        # Initialize embedder
        logger.info(f"Initializing sentence transformer: {self.text_encoder}")
        encoder = self._get_text_encoder()
        
        # Batch process embeddings for all unique questions
        batch_size = 10000  # Adjust based on GPU memory
        logger.info(f"Processing {len(unique_questions)} unique questions in batches of {batch_size}...")
        all_embeddings = []
        
        for i in range(0, len(unique_questions), batch_size):
            batch_prompts = unique_questions['prompt'].iloc[i:i+batch_size].tolist()
            batch_embeddings = encoder.encode(batch_prompts, show_progress_bar=True, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Stack all embeddings
        embeddings_array = np.vstack(all_embeddings)
        
        # Create labels matrix (model_num, query_num) where values are actual labels
        logger.info("Creating labels matrix...")
        labels_matrix = df.pivot(index='model_id', columns='prompt_id', values='label').fillna(0).astype(int)
        labels_array = labels_matrix.values
        
        # Create embeddings tensor corresponding to query_num dimension
        # Map prompt_ids to their embedding indices
        prompt_id_to_embedding_idx = {prompt_id: idx for idx, prompt_id in enumerate(unique_questions['prompt_id'])}
        
        # Get embeddings in the same order as the labels matrix columns
        embedding_indices = [prompt_id_to_embedding_idx[prompt_id] for prompt_id in labels_matrix.columns]
        ordered_embeddings = embeddings_array[embedding_indices]
        
        # Create text list in the same order as embeddings
        prompt_id_to_text = {prompt_id: text for prompt_id, text in zip(unique_questions['prompt_id'], unique_questions['prompt'])}
        text_list = [prompt_id_to_text[prompt_id] for prompt_id in labels_matrix.columns]
        
        # Convert to tensors
        labels_tensor = torch.tensor(labels_array, dtype=torch.float32)
        embeddings_tensor = torch.tensor(ordered_embeddings, dtype=torch.float32)
        
        logger.info(f"Created encoded data:")
        logger.info(f"  Labels shape: {labels_tensor.shape}")
        logger.info(f"  Embeddings shape: {embeddings_tensor.shape}")
        logger.info(f"  Text list length: {len(text_list)}")
        logger.info(f"Labels matrix shape: (models={labels_tensor.shape[0]}, queries={labels_tensor.shape[1]})")
        logger.info(f"Embeddings tensor shape: (queries={embeddings_tensor.shape[0]}, embedding_dim={embeddings_tensor.shape[1]})")
        
        return labels_tensor, embeddings_tensor, text_list
    
    def load_encoded_data(self, data_type: str = "train") -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Load pre-encoded data from the text encoder specific directory.
        If encoded data doesn't exist, create it from raw CSV files.
        
        Args:
            data_type: 'train', 'val', or 'test'
            
        Returns:
            tuple: (labels_tensor, embeddings_tensor, text_list)
        """
        encoded_path = f"{self.encoded_dir}/{data_type}"
        
        # Check if encoded data exists
        labels_path = f"{encoded_path}/labels_tensor.pt"
        embeddings_path = f"{encoded_path}/embeddings_tensor.pt"
        text_list_path = f"{encoded_path}/text_list.pkl"
        
        if not all(os.path.exists(p) for p in [labels_path, embeddings_path, text_list_path]):
            logger.info(f"Encoded data not found for {data_type}, creating from raw CSV...")
            
            # Create encoded data from raw CSV
            labels_tensor, embeddings_tensor, text_list = self.create_encoded_data_from_raw(data_type)
            
            # Save the encoded data for future use
            self.save_encoded_data(labels_tensor, embeddings_tensor, text_list, data_type)
            
            return labels_tensor, embeddings_tensor, text_list
        
        # Load existing encoded data
        labels_tensor = torch.load(labels_path, map_location=torch.device("cpu"))
        embeddings_tensor = torch.load(embeddings_path, map_location=torch.device("cpu"))
        
        with open(text_list_path, "rb") as f:
            text_list = pickle.load(f)
        
        logger.info(f"Loaded {data_type} encoded data:")
        logger.info(f"  Labels shape: {labels_tensor.shape}")
        logger.info(f"  Embeddings shape: {embeddings_tensor.shape}")
        logger.info(f"  Text list length: {len(text_list)}")
        
        return labels_tensor, embeddings_tensor, text_list
    
    def prepare_training_data(self, 
                             labels_tensor: torch.Tensor, 
                             embeddings_tensor: torch.Tensor, 
                             text_list: List[str],
                             batch_size: int = 32,
                             shuffle: bool = True) -> Dict[str, DataLoader]:
        """
        Prepare data for training with multi-output MLP
        
        Args:
            labels_tensor: (n_models, n_queries) tensor of labels
            embeddings_tensor: (n_queries, embedding_dim) tensor of embeddings
            text_list: List of text prompts
            batch_size: Batch size for data loaders
            shuffle: Whether to shuffle training data
            
        Returns:
            dict: Dictionary containing training data loader
        """
        logger.info("Preparing data for training...")
        if self.model_type == "mirt":
            return self._prepare_pairwise_training_data(
                labels_tensor=labels_tensor,
                embeddings_tensor=embeddings_tensor,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        
        n_models, n_queries = labels_tensor.shape
        embedding_dim = embeddings_tensor.shape[1]
        
        logger.info(f"Data shapes: {n_models} models, {n_queries} queries, {embedding_dim} embedding dim")
        
        # For multi-output MLP: each query gets all model predictions
        # X: (n_queries, embedding_dim) - one embedding per query
        # y: (n_queries, n_models) - all model predictions for each query
        X = embeddings_tensor  # (n_queries, embedding_dim)
        y = labels_tensor.T    # (n_queries, n_models) - transpose to get queries x models
        
        logger.info(f"Multi-output data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Each sample: embedding_dim={embedding_dim}, n_models={n_models}")
        
        # Create data loader
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        logger.info(f"Created training data loader with batch size: {batch_size}")
        logger.info(f"Each batch will have shape: X=(batch_size, {embedding_dim}), y=(batch_size, {n_models})")
        
        return {
            'train_loader': train_loader,
            'train_dataset': train_dataset,
            'n_models': n_models,
            'embedding_dim': embedding_dim
        }
    
    def prepare_test_data(self, 
                         labels_tensor: torch.Tensor, 
                         embeddings_tensor: torch.Tensor, 
                         text_list: List[str],
                         batch_size: int = 32) -> DataLoader:
        """
        Prepare test data for evaluation with multi-output MLP
        
        Args:
            labels_tensor: (n_models, n_queries) tensor of labels
            embeddings_tensor: (n_queries, embedding_dim) tensor of embeddings
            text_list: List of text prompts
            batch_size: Batch size for data loader
            
        Returns:
            DataLoader for test data
        """
        logger.info("Preparing test data...")
        if self.model_type == "mirt":
            return self._prepare_pairwise_test_loader(
                labels_tensor=labels_tensor,
                embeddings_tensor=embeddings_tensor,
                batch_size=batch_size,
            )
        
        n_models, n_queries = labels_tensor.shape
        embedding_dim = embeddings_tensor.shape[1]
        
        logger.info(f"Test data shapes: {n_models} models, {n_queries} queries, {embedding_dim} embedding dim")
        
        # For multi-output MLP: each query gets all model predictions
        # X: (n_queries, embedding_dim) - one embedding per query
        # y: (n_queries, n_models) - all model predictions for each query
        X = embeddings_tensor  # (n_queries, embedding_dim)
        y = labels_tensor.T    # (n_queries, n_models) - transpose to get queries x models
        
        logger.info(f"Multi-output test data shape: X={X.shape}, y={y.shape}")
        
        # Create test dataset and loader
        test_dataset = TensorDataset(X, y)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created test data loader with batch size: {batch_size}")
        logger.info(f"Each batch will have shape: X=(batch_size, {embedding_dim}), y=(batch_size, {n_models})")
        
        return test_loader

    def _get_pairwise_tensors(
        self,
        labels_tensor: torch.Tensor,
        embeddings_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert matrix-form labels and embeddings into pairwise tensors.
        """
        cache_key = (labels_tensor.data_ptr(), embeddings_tensor.data_ptr())
        if cache_key in self._pairwise_cache:
            return self._pairwise_cache[cache_key]

        labels_tensor = labels_tensor.contiguous()
        embeddings_tensor = embeddings_tensor.contiguous()

        n_models, n_queries = labels_tensor.shape
        embedding_dim = embeddings_tensor.shape[1]

        if self.llm_embeddings is None:
            raise ValueError(
                "LLM embeddings are not set. Call `set_llm_embeddings` with a "
                f"(n_models={n_models}, embedding_dim={embedding_dim}) tensor before using the MIRT pipeline."
            )
        if self.llm_embeddings.shape != (n_models, embedding_dim):
            raise ValueError(
                "LLM embeddings shape mismatch. Expected "
                f"({n_models}, {embedding_dim}), got {self.llm_embeddings.shape}."
            )

        llm_features = self.llm_embeddings.repeat_interleave(n_queries, dim=0)
        query_embeddings = embeddings_tensor.repeat(n_models, 1)
        responses = labels_tensor.reshape(-1)

        pairwise = (llm_features, query_embeddings, responses)
        self._pairwise_cache[cache_key] = pairwise
        return pairwise

    def _prepare_pairwise_training_data(
        self,
        labels_tensor: torch.Tensor,
        embeddings_tensor: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Dict[str, DataLoader]:
        """
        Prepare pairwise (llm, query) data suitable for MIRT models.
        """
        logger.info("Preparing pairwise data for MIRT...")

        llm_features, query_embeddings, responses = self._get_pairwise_tensors(labels_tensor, embeddings_tensor)

        n_models = labels_tensor.shape[0]
        embedding_dim = embeddings_tensor.shape[1]

        logger.info(f"Pairwise data shapes: {n_models} models, embedding_dim={embedding_dim}, total_pairs={len(llm_features)}")

        dataset = TensorDataset(llm_features, query_embeddings, responses)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return {
            'train_loader': loader,
            'train_dataset': dataset,
            'n_models': n_models,
            'embedding_dim': embedding_dim
        }

    def _prepare_pairwise_test_loader(
        self,
        labels_tensor: torch.Tensor,
        embeddings_tensor: torch.Tensor,
        batch_size: int = 32,
    ) -> DataLoader:
        """
        Prepare pairwise data loader for validation or testing.
        """
        llm_features, query_embeddings, responses = self._get_pairwise_tensors(labels_tensor, embeddings_tensor)

        dataset = TensorDataset(llm_features, query_embeddings, responses)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def save_encoded_data(self, 
                         labels_tensor: torch.Tensor, 
                         embeddings_tensor: torch.Tensor, 
                         text_list: List[str],
                         data_type: str = "train") -> None:
        """
        Save encoded data to the text encoder specific directory
        
        Args:
            labels_tensor: (n_models, n_queries) tensor of labels
            embeddings_tensor: (n_queries, embedding_dim) tensor of embeddings
            text_list: List of text prompts
            data_type: 'train' or 'test'
        """
        # Create directory if it doesn't exist
        save_dir = f"{self.encoded_dir}/{data_type}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save tensors
        torch.save(labels_tensor, f"{save_dir}/labels_tensor.pt")
        torch.save(embeddings_tensor, f"{save_dir}/embeddings_tensor.pt")
        
        # Save text list
        with open(f"{save_dir}/text_list.pkl", "wb") as f:
            pickle.dump(text_list, f)
        
        # Save encoder info
        encoder_info = {
            "text_encoder": self.text_encoder,
            "encoder_name": self.encoder_name,
            "data_type": data_type,
            "n_models": labels_tensor.shape[0],
            "n_queries": labels_tensor.shape[1],
            "embedding_dim": embeddings_tensor.shape[1]
        }
        
        with open(f"{save_dir}/encoder_info.json", "w") as f:
            json.dump(encoder_info, f, indent=2)
        
        logger.info(f"Saved encoded {data_type} data to {save_dir}")
        logger.info(f"Encoder info: {encoder_info}")


def get_data_processor(config: dict):
    """
    Get a data processor from configuration dictionary
    
    Args:
        config: Configuration dictionary containing data_config and training_config
        
    Returns:
        Data processor instance based on dataset name
    """
    data_config = config['data_config']
    training_config = config['training_config']
    pipeline_config = config['pipeline_config']
    
    dataset_name = data_config['dataset_name']
    text_encoder = data_config['text_encoder']
    device = training_config['device']
    model_type = pipeline_config.get('model_type', 'mlp')
    
    # Return appropriate processor based on dataset name
    if dataset_name == "EmbedLLM":
        return EmbedLLMDataProcessor(
            dataset_name=dataset_name,
            text_encoder=text_encoder,
            device=device,
            model_type=model_type
        )
    elif dataset_name == "RouterBench":
        return RouterBenchDataProcessor(
            dataset_name=dataset_name,
            text_encoder=text_encoder,
            device=device,
            model_type=model_type
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Currently supported: 'EmbedLLM', 'RouterBench'")