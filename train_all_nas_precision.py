import subprocess
def train_command(scheme, id):
  return ['python','/root/workspace/models/research/slim/train_image_classifier.py','--train_dir=/tmp/nasnet_mobile/'+str(id)+'/trainlogs','--dataset_name=imagenet','--dataset_split_name=train','--dataset_dir=/root/workspace/data/imagenet','--checkpoint_path=/root/workspace/nasnet/checkpoint/model.ckpt','--ignore_missing_vars','--model_name=nasnet_mobile','--max_number_of_steps=500','--train_image_size=224','--learning_rate=0.00001','--weight_decay=0.00004','--ignore_missing_vars','--quantize_delay=0','--quantization_list='+scheme]
for i in range(6,17):
  f = open("train_logs/logs_"+str(i) , "w+" )
  sch = ""
  for j in range(198):
    sch = sch + str(i) + ","
  subprocess.call(train_command(sch[:-1],i),stderr=f)
  f.close()
 
