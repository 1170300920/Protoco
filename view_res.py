import re
import matplotlib.pyplot as plt

datasets=['scifact','fever','vc']
shots=[3,6,12,24,48]
K=[1,2,4,8,16]
seeds=[0,2,3,4]
filenames=[]
f1s=[]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for dataset in datasets:
    f1_seeds=[]
    for seed in seeds:             
        f1_shots=[]
        for sh in shots:
            #output/vc_shots48_seed3_fs_stage2/test_pred.txt
            filename="output/{}_shots{}_seed{}_fs_stage2/test_pred.txt".format(dataset,sh,seed)
            filenames.append(filename)
            try:
                with open(filename,'r') as f:
                    line=f.readlines()[0]
                    match=re.search(r"Test F1 : (\d+\.\d+)",line)
                    if match:
                        f1=float(match.group(1))
                        f1s.append(f1)
                        f1_shots.append(f1)
                    else:
                        print(f"F1 value not found in the {filename}.",filename)
                        f1s.append(0)
                        f1_shots.append(0)
                        continue
            except FileNotFoundError:
                print(f" not found  {filename}.",filename)
                f1s.append(0)
                f1_shots.append(0)
                continue
        f1_seeds.append(f1_shots)
    print(f1_seeds)
    axes[datasets.index(dataset)].set_title(dataset)
    axes[datasets.index(dataset)].plot(K,f1_seeds[0],label="seed0",marker='o')
    axes[datasets.index(dataset)].plot(K,f1_seeds[1],label="seed2",marker='o')
    axes[datasets.index(dataset)].plot(K,f1_seeds[2],label="seed3",marker='o')
    axes[datasets.index(dataset)].plot(K,f1_seeds[3],label="seed4",marker='o')
    plt.xlabel("k")
    plt.ylabel("F1")
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("f1.png")
    

with open("output_f1.txt",'w') as outfile:
    for i in range(len(filenames)):
        outfile.write("{}\t{}\n".format(filenames[i],f1s[i]))