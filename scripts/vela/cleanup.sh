#/bin/bash

job_name=${1}

oc delete appwrapper ${job_name}
oc delete configmap ${job_name}-config
