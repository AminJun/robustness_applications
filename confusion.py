from soft_xent import SoftCrossEntropy, CachedData
import sys

GLOBAL_MODE = int(sys.argv[1])

cached_data = CachedData('.', GLOBAL_MODE)
cached_data.cache(None, None, None)
label = cached_data().cuda()
print(label)
