# Run experiments on SDD
# Author: Ryo Yonetani
# Affiliation: OMRON SINIC X
# 
# usage:
# $ sh 3_SDD.sh | sh (if you want to run experiments sequentially)

logdir=icml2021_sdd
save_dir=log/icml2021_sdd

for test_scene in video0 bookstore coupa deathCircle gates hyang little nexus quad
do
    echo python train_sdd.py -m NeuralAstar -t $test_scene -s $save_dir -d data/sdd/s064_0.5_128_300 \
    -gf config/macros.gin config/neural_astar.gin config/bb_astar.gin
    echo python train_sdd.py -m BBAstar -t $test_scene -s $save_dir -d data/sdd/s064_0.5_128_300 \
    -gf config/macros.gin config/neural_astar.gin config/bb_astar.gin
done