# Pytorch Lightning Demo
Demo codes for using Pytorch Lightning with Slurm and TensorBoard

## Start training
<pre>
sbatch scripts/train.sh
</pre>

## Monitoring with TensorBoard
<pre>
tensorboard --logdir logs/lightning_logs/
</pre>