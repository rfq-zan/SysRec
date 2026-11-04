import numpy as np

bge_file = 'cache_embedding/bge_corpus_3583fe1c97.npy'
e5_file = 'cache_embedding/e5_corpus_8ded02df08.npy'

for file in [bge_file, e5_file]:
    print(f"Splitting {file}...")
    arr = np.load(file)
    mid = len(arr) // 2
    np.save(file.replace('.npy', '_pt1.npy'), arr[:mid])
    np.save(file.replace('.npy', '_pt2.npy'), arr[mid:])
    
print("Done! You now have 65MB chunks.")