import numpy as np

# Make sure the folder name exactly matches what's on your machine
bge_file = 'cache_embedding/bge_corpus_3583fe1c97.npy'
e5_file = 'cache_embedding/e5_corpus_8ded02df08.npy'

for file in [bge_file, e5_file]:
    print(f"Splitting {file} into 3 parts...")
    arr = np.load(file)
    third = len(arr) // 3
    
    np.save(file.replace('.npy', '_pt1.npy'), arr[:third])
    np.save(file.replace('.npy', '_pt2.npy'), arr[third:third*2])
    np.save(file.replace('.npy', '_pt3.npy'), arr[third*2:])
    
print("Done! You now have 43MB chunks (Server safe).")