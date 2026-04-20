
# models.py contains a single dataclass, JobState, which groups together
# a set of filesystem paths and little state (completed steps, errors).
# It provides a convenience property is_done that returns True when the
# discovered raw job plus the generated outputs exist on disk.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class JobState:
    folder: Path
    raw_file: Path
    metadata_file: Path
    cleaned_file: Path
    pdf_file: Path
    info_file: Path
    completed_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def is_done(self) -> bool:
        return (
            self.raw_file.exists()
            and self.cleaned_file.exists()
            and self.pdf_file.exists()
            and self.info_file.exists()
        )
