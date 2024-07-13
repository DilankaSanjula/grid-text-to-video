# grid-text-to-video
MSc Research UOM


# Version 1
## Basic text to image generation using stable diffusion

## Special steps to intialize project

Fix for:  ModuleNotFoundError: No module named 'resource'

Add the following to shuffle.py in lib "/site-packages/tensorflow_datasets/core/shuffle.py"

```bash
import os

try:
    if os.name != 'nt':
        import resource
except ImportError:
    # resource module is not available on Windows
    pass
```
