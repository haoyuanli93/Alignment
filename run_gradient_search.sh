bsub -q psanaq -n 24 -R"span[ptile=1]" -o %J.out mpirun --mca btl ^openib \
python /reg/neh/home5/haoyuan/Documents/my_repos/Alignment/gradient_search.py \
--output "/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output" \
--fixed_target "/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/input/chop_complete.npy" \
--movable_target "/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/input/chop_category_2.npy" \
--iter_num 9 \
--tag "cat_2"
