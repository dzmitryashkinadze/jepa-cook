import polars as pl
import torch
from torch.utils.data import Dataset


class PreTokenizedActionDataset(Dataset):
    """Dataset class for structural pre-tokenized recipes."""

    def __init__(self, dataset_path: str) -> None:
        """Initializes the dataset by reading a Parquet file natively.

        Args:
            dataset_path: Path to the structural parquet data file.
        """
        self.df: pl.DataFrame = pl.read_parquet(dataset_path)

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Fetches a single recipe row and casts item arrays to long tensors.

        Args:
            idx: Row index.

        Returns:
            A tuple containing structural context x, actions a, and target labels y.
        """
        row = self.df.row(idx, named=True)
        x = [torch.tensor(item, dtype=torch.long) for item in row["x_tokens"]]
        a = [torch.tensor(item, dtype=torch.long) for item in row["a_tokens"]]
        y = torch.tensor(row["y_tokens"], dtype=torch.long)
        return x, a, y


def pad_nested_sequences(batch_lists: list[list[torch.Tensor]], max_len: int = 128) -> torch.Tensor:
    """Pads a nested batch of varied length structural elements into a 3D Tensor.

    Args:
        batch_lists: A batch list of lists containing variable-length 1D tensors.
        max_len: Target sequence length threshold to pad or clip.

    Returns:
        Padded tensor layout of shape [batch_size, max_elements_in_batch, max_len].
    """
    batch_size: int = len(batch_lists)
    max_elements: int = max(len(row) for row in batch_lists)
    max_elements = max(1, max_elements)

    padded_tensor: torch.Tensor = torch.zeros(batch_size, max_elements, max_len, dtype=torch.long)

    for i, row in enumerate(batch_lists):
        for j, element in enumerate(row):
            length = min(len(element), max_len)
            if length > 0:
                padded_tensor[i, j, :length] = element[:length]

    return padded_tensor


class JEPACollateFn:
    """Collator object to pad varying lengths within structural recipe steps dynamically."""

    def __init__(self, max_len: int = 128) -> None:
        """Initializes the collator configuration.

        Args:
            max_len: System-wide uniform token layout boundary limit.
        """
        self.max_len: int = max_len

    def __call__(
        self, batch: list[tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pads structural layers dynamically inside current mini-batch runtime.

        Args:
            batch: Batch collection returned from the dataset pipeline.

        Returns:
            Tensors of structural layouts: x, a, and target sequence y.
        """
        xs, as_, ys = zip(*batch)

        x_tensor: torch.Tensor = pad_nested_sequences(xs, max_len=self.max_len)
        a_tensor: torch.Tensor = pad_nested_sequences(as_, max_len=self.max_len)

        y_tensor: torch.Tensor = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
        if y_tensor.size(1) < self.max_len:
            padding = torch.zeros(y_tensor.size(0), self.max_len - y_tensor.size(1), dtype=torch.long)
            y_tensor = torch.cat([y_tensor, padding], dim=1)
        else:
            y_tensor = y_tensor[:, : self.max_len]

        return x_tensor, a_tensor, y_tensor
