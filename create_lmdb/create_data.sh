cafferoot_dir=$HOME/Downloads/caffe
cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir

cd $root_dir
export PYTHONPATH=$PYTHONPATH:$cafferoot_dir/python

redo=1
data_root_dir="$HOME/Downloads/Mask"
dataset_name="VOC2007"
mapfile="$cur_dir/$dataset_name/labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=300
height=300

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python3 $cafferoot_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $cur_dir/$subset.txt $cur_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
