"""Native desktop UI for the research sandbox.

Import-order workaround: pandas must be imported before PySide6. Qt pulls in a
``six`` meta-path importer, and once that is installed pandas' Cython extension
modules fail to import with a confusing ``_SixMetaPathImporter`` error. Doing it
here means any module under ``app`` gets the safe order for free.
"""

import pandas as _pandas  # noqa: F401  (import order matters - see above)
