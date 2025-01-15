# Vela Instructions

## Submit jobs using the submit_to_vela script

1. Update an existing training configuration file to have a dedicated vela part. See [config_example.sh](config_example.sh) for reference.
2. ```bash submit_to_vela.sh <path/to/your/config/file.yaml>```
3. Check job status using oc get appwrapper `<job-name>`  and logs OpenShift console or `oc get logs <pod-name>`
4. Once the job is finished cleanup: ```bash cleanup.sh <job-name>```

## General instructions on how to prepare and use Vela

For more information see https://github.ibm.com/app-modernization/vela-instructions 

### Instructions to Prepare for Vela

Everyone must complete the one time steps in the section [Preparing for Vela](https://github.ibm.com/app-modernization/vela-instructions/blob/main/docs/preparing-for-vela.md). Once these are completed, you are ready to submit jobs.

### Instructions for using Vela

For general information about how to interact with the environment [Working with Vela](https://github.ibm.com/app-modernization/vela-instructions/blob/main/docs/working-with-vela.md).

### Prepare requirements for vela

To export `requirements.txt` for vela job submission run:

```console
uv export --extras modeling --format requirements-txt --no-hashes --no-urls > requirements.txt
```
