#/bin/bash

SCRIPT_DIR_PATH=$(dirname "$0")

configuration_path=${1}
vela_template_configuration_file_path=${2-vela_template.yaml}
helm_template_path=${3-fms-job-templates}

job_file_path=$(python vela_preparation.py ${configuration_path} ${vela_template_configuration_file_path} ${helm_template_path}/enqueue-job/chart/files)

helm template -f ${job_file_path} ${helm_template_path}/enqueue-job/chart | oc create -f -

rm ${job_file_path}
rm ${helm_template_path}/enqueue-job/chart/files/${job_file_path}
