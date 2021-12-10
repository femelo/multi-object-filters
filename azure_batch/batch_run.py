#!/usr/bin/env python3
from __future__ import print_function
import datetime
import os
import sys
import time
import yaml
from copy import copy
from natsort import natsorted
from math import floor, log10
sys.path.append('.')
sys.path.append('..')
sys.path.append('azure_batch')

CONFIG_FILE = 'config.yaml'

# Load parameters
FOLDERS = ['.', '..', 'azure_batch']
config_file_found = False
for folder in FOLDERS:
    path_to_config_file = os.path.abspath(os.path.join(folder, CONFIG_FILE))
    if os.path.exists(path_to_config_file):
        config_file_found = True
        break
if config_file_found:
    with open(path_to_config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
else:
    print('Could not find the configuration file (yaml). Please make sure the file is in the repository folder.')
    exit()

try:
    input = raw_input
except NameError:
    pass

import azure.storage.blob as azure_blob
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as batch_models
import azure.mgmt.batch as batch_management
import azure.identity as identity

def get_latest_app_version(batch_mgmt_client, app_id):
    """
    Given the name of the batch application, returns the latest version in use.
    Note: The assumption here is that the latest version is the one set as the default version. Natsort will sort the
    versions regardless of most formats, and return the ordered  list.
    :param batch_client: the batch client
    :param app_id: the name of the application
    :return: the latest version present in the list of application packages for the application
    """
    iterator = batch_mgmt_client.application_package.list(
        resource_group_name=config['RESOURCE_GROUP_NAME'], 
        account_name=config['BATCH_ACCOUNT_NAME'],
        application_name=app_id
    )
    versions = []
    while True:
        try:
            versions.append(iterator.next().name)
        except StopIteration:
            break
    if len(versions) > 0:
        latest_version = natsorted(versions)[-1]
    else:
        latest_version = None
    return latest_version

# Update the Batch and Storage account credential strings in config.yaml with values
# unique to your accounts. These are used when constructing connection strings
# for the Batch and Storage client objects.
def query_yes_no(question, default="yes"):
    """
    Prompts the user for yes/no input, displaying the specified question text.

    :param str question: The text of the prompt for input.
    :param str default: The default if the user hits <ENTER>. Acceptable values
    are 'yes', 'no', and None.
    :rtype: str
    :return: 'yes' or 'no'
    """
    valid = {'y': 'yes', 'n': 'no'}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("Invalid default answer: '{}'".format(default))

    while 1:
        choice = input(question + prompt).lower()
        if default and not choice:
            return default
        try:
            return valid[choice[0]]
        except (KeyError, IndexError):
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def print_batch_exception(batch_exception):
    """
    Prints the contents of the specified Batch exception.

    :param batch_exception:
    """
    print('-------------------------------------------')
    print('Exception encountered:')
    if batch_exception.error and \
            batch_exception.error.message and \
            batch_exception.error.message.value:
        print(batch_exception.error.message.value)
        if batch_exception.error.values:
            print()
            for mesg in batch_exception.error.values:
                print('{}:\t{}'.format(mesg.key, mesg.value))
    print('-------------------------------------------')

def upload_file_to_container(block_blob_client, container_name, file_path):
    """
    Uploads a local file to an Azure Blob storage container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param str file_path: The local path to the file.
    :rtype: `azure.batch.models.ResourceFile`
    :return: A ResourceFile initialized with a SAS URL appropriate for Batch
    tasks.
    """
    blob_name = os.path.basename(file_path)

    print('Uploading file {} to container [{}]...'.format(file_path,
                                                          container_name))

    block_blob_client.create_blob_from_path(container_name,
                                            blob_name,
                                            file_path)

    # Obtain the SAS token for the container.
    sas_token = get_container_sas_token(block_blob_client,
                                        container_name, azure_blob.BlobPermissions.READ)

    sas_url = block_blob_client.make_blob_url(container_name,
                                              blob_name,
                                              sas_token=sas_token)

    return batch_models.ResourceFile(file_path=blob_name,
                                    http_url=sas_url)

def get_container_sas_token(block_blob_client,
                            container_name, blob_permissions):
    """
    Obtains a shared access signature granting the specified permissions to the
    container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param BlobPermissions blob_permissions:
    :rtype: str
    :return: A SAS token granting the specified permissions to the container.
    """
    # Obtain the SAS token for the container, setting the expiry time and
    # permissions. In this case, no start time is specified, so the shared
    # access signature becomes valid immediately. Expiration is in 2 hours.
    container_sas_token = \
        block_blob_client.generate_container_shared_access_signature(
            container_name,
            permission=blob_permissions,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2))

    return container_sas_token

def get_container_sas_url(block_blob_client,
                          container_name, blob_permissions):
    """
    Obtains a shared access signature URL that provides write access to the 
    ouput container to which the tasks will upload their output.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param BlobPermissions blob_permissions:
    :rtype: str
    :return: A SAS URL granting the specified permissions to the container.
    """
    # Obtain the SAS token for the container.
    sas_token = get_container_sas_token(block_blob_client,
                                        container_name, azure_blob.BlobPermissions.WRITE)

    # Construct SAS URL for the container
    container_sas_url = "https://{}.blob.core.windows.net/{}?{}".format(
        config['STORAGE_ACCOUNT_NAME'], container_name, sas_token)

    return container_sas_url

def create_pool(batch_service_client, pool_id, app_id, app_version):
    """
    Creates a pool of compute nodes with the specified OS settings.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str pool_id: An ID for the new pool.
    :param str publisher: Marketplace image publisher
    :param str offer: Marketplace image offer
    :param str sku: Marketplace image sky
    """
    print('Creating pool [{}]...'.format(pool_id))

    # Create a new pool of Linux compute nodes using an Azure Virtual Machines
    # Marketplace image. For more information about creating pools of Linux
    # nodes, see:
    # https://azure.microsoft.com/documentation/articles/batch-linux-nodes/
    new_pool = batch.models.PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=batch_models.VirtualMachineConfiguration(
            image_reference=batch_models.ImageReference(
                publisher="Canonical",
                offer="UbuntuServer",
                sku="18.04-LTS",
                version="latest"
            ),
            node_agent_sku_id="batch.node.ubuntu 18.04"),
        vm_size=config['POOL_VM_SIZE'],
        target_dedicated_nodes=config['DEDICATED_POOL_NODE_COUNT'],
        target_low_priority_nodes=config['LOW_PRIORITY_POOL_NODE_COUNT'],
        application_package_references=[
            batch_models.ApplicationPackageReference(
                application_id=app_id, version=app_version
            )
        ],
    )

    create_pool_if_not_exist(batch_service_client, new_pool)

    # because we want all nodes to be available before any tasks are assigned
    # to the pool, here we will wait for all compute nodes to reach idle
    nodes = wait_for_all_nodes_state(
        batch_service_client, new_pool,
        frozenset(
            (batch_models.ComputeNodeState.start_task_failed,
             batch_models.ComputeNodeState.unusable,
             batch_models.ComputeNodeState.idle)
        )
    )
    # ensure all node are idle
    if any(node.state != batch_models.ComputeNodeState.idle for node in nodes):
        raise RuntimeError('node(s) of pool {} not in idle state'.format(
            pool_id))

def create_job(batch_service_client, job_id, pool_id):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID for the job.
    :param str pool_id: The ID for the pool.
    """
    print('Creating job [{}]...'.format(job_id))

    job = batch.models.JobAddParameter(
        id=job_id,
        pool_info=batch.models.PoolInformation(pool_id=pool_id))

    batch_service_client.job.add(job)

def add_tasks(batch_service_client, job_id, num_of_tasks, output_container_sas_url, app_id, app_version):
    """
    Adds a task for each input file in the collection to the specified job.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID of the job to which to add the tasks.
    :param num_of_tasks: Number of tasks to be created.
    :param output_container_sas_token: A SAS token granting write access to
    the specified Azure Blob storage container.
    """

    print('Adding {} tasks to job [{}]...'.format(num_of_tasks, job_id))
    tasks = list()

    # Get remote app path
    app_id_ = copy(app_id)
    app_version_ = copy(app_version)
    for c in ['.', '-', '#']:
        if c in app_id_:
            app_id_ = app_id_.replace(c, '_')
        if c in app_version_:
            app_version_ = app_version_.replace(c, '_')
    remote_app_path = '/'.join(
        [
            '_'.join(['$AZ_BATCH_APP_PACKAGE', app_id_, app_version_]),
            config['APP_BINARY']
        ]
    )

    d = floor(log10(num_of_tasks)) + 1
    for idx in range(num_of_tasks):
        output_file_name = config['OUTPUT_FILENAME'].format(idx)
        output_file_path = os.path.join(config['OUTPUT_DIR'], output_file_name)
        # Set full command
        app_command = ' '.join([remote_app_path, config['APP_ARGS'], '-s', '-o {}'.format(output_file_name)])
        command = "/bin/bash -c \"{} \"".format(app_command)
        tasks.append(batch.models.TaskAddParameter(
            id='Task-{{:0{:d}}}'.format(d).format(idx),
            command_line=command,
            output_files=[batch_models.OutputFile(
                file_pattern=output_file_path,
                destination=batch_models.OutputFileDestination(
                          container=batch_models.OutputFileBlobContainerDestination(
                              container_url=output_container_sas_url)),
                upload_options=batch_models.OutputFileUploadOptions(
                    upload_condition=batch_models.OutputFileUploadCondition.task_success))]
        )
        )
    batch_service_client.task.add_collection(job_id, tasks)

def create_pool_if_not_exist(batch_client, pool):
    """Creates the specified pool if it doesn't already exist

    :param batch_client: The batch client to use.
    :type batch_client: `batchserviceclient.BatchServiceClient`
    :param pool: The pool to create.
    :type pool: `batchserviceclient.models.PoolAddParameter`
    """
    try:
        print("Attempting to create pool:", pool.id)
        batch_client.pool.add(pool)
        print("Created pool:", pool.id)
    except batch_models.BatchErrorException as e:
        if e.error.code != "PoolExists":
            raise
        else:
            print("Pool {!r} already exists".format(pool.id))

def wait_for_all_nodes_state(batch_client, pool, node_state):
    """Waits for all nodes in pool to reach any specified state in set

    :param batch_client: The batch client to use.
    :type batch_client: `batchserviceclient.BatchServiceClient`
    :param pool: The pool containing the node.
    :type pool: `batchserviceclient.models.CloudPool`
    :param set node_state: node states to wait for
    :rtype: list
    :return: list of `batchserviceclient.models.ComputeNode`
    """
    print('Waiting for all nodes in pool {} to reach one of: {!r}...'.format(
        pool.id, node_state))
    start_time = datetime.datetime.now()
    while True:
        # refresh pool to ensure that there is no resize error
        pool = batch_client.pool.get(pool.id)
        if pool.resize_errors is not None:
            resize_errors = "\n".join([repr(e) for e in pool.resize_errors])
            raise RuntimeError(
                'resize error encountered for pool {}:\n{}'.format(
                    pool.id, resize_errors))
        nodes = list(batch_client.compute_node.list(pool.id))
        in_desired_state = [node.state in node_state for node in nodes]
        n_all = len(nodes)
        n_ok = len([node_ok for node_ok in in_desired_state if node_ok])
        print('\rWaiting: {:03d} / {:03d} nodes in desired state... {} '.format(
                n_ok, n_all, str(datetime.datetime.now() - start_time).split('.', 2)[0]), end='')
        if (len(nodes) >= pool.target_dedicated_nodes and
                all(in_desired_state)):
            print('done.')
            return nodes
        time.sleep(1)

def wait_for_tasks_to_complete(batch_service_client, job_id, timeout):
    """
    Returns when all tasks in the specified job reach the Completed state.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The id of the job whose tasks should be monitored.
    :param timedelta timeout: The duration to wait for task completion. If all
    tasks in the specified job do not reach Completed state within this time
    period, an exception will be raised.
    """
    timeout_expiration = datetime.datetime.now() + timeout

    print("Monitoring all tasks for 'Completed' state, timeout in {}..."
          .format(timeout), end='')

    print()
    while datetime.datetime.now() < timeout_expiration:
        print('\rTime remaining: {}'.format(str(timeout_expiration - datetime.datetime.now()).split('.', 2)[0]), end='')
        sys.stdout.flush()
        tasks = batch_service_client.task.list(job_id)

        incomplete_tasks = [task for task in tasks if
                            task.state != batch_models.TaskState.completed]
        if not incomplete_tasks:
            print()
            return True
        else:
            time.sleep(1)

    print()
    raise RuntimeError("ERROR: Tasks did not reach 'Completed' state within "
                       "timeout period of " + str(timeout))

if __name__ == '__main__':

    start_time = datetime.datetime.now().replace(microsecond=0)
    print('Sample start: {}'.format(start_time))
    print()

    # Log in
    credential = identity.AzureCliCredential()
    # Create batch management client
    batch_mgmt_client = batch_management.BatchManagementClient(
        credential = credential, subscription_id=config['SUBSCRIPTION_ID'])
    app_id = config['APP_ID']
    app_version = get_latest_app_version(batch_mgmt_client, app_id)
    if app_version is None:
        print('No version of [{}] has been found in Azure. Please deploy the application first.'.format(app_id))
        exit()
    print('Job will use the latest version of the application [{}-{}].'.format(app_id, app_version))

    # Create the blob client, for use in obtaining references to
    # blob storage containers and uploading files to containers.
    blob_client = azure_blob.BlockBlobService(
        account_name=config['STORAGE_ACCOUNT_NAME'],
        account_key=config['STORAGE_ACCOUNT_KEY'])

    # Use the blob client to create the containers in Azure Storage if they
    # don't yet exist.
    output_container_name = 'output'
    blob_client.create_container(output_container_name, fail_on_exist=False)
    print('Container [{}] created.'.format(output_container_name))

    num_of_tasks = config['NUMBER_OF_TASKS']

    # Obtain a shared access signature URL that provides write access to the output
    # container to which the tasks will upload their output.
    output_container_sas_url = get_container_sas_url(
        blob_client,
        output_container_name,
        azure_blob.BlobPermissions.WRITE)

    # Create a Batch service client. We'll now be interacting with the Batch
    # service in addition to Storage
    credentials = batch_auth.SharedKeyCredentials(config['BATCH_ACCOUNT_NAME'],
                                                 config['BATCH_ACCOUNT_KEY'])

    batch_client = batch.BatchServiceClient(
        credentials,
        batch_url=config['BATCH_ACCOUNT_URL'])

    try:
        # Create the pool that will contain the compute nodes that will execute the
        # tasks.
        create_pool(batch_client, config['POOL_ID'], app_id, app_version)

        # Create the job that will run the tasks.
        create_job(batch_client, config['JOB_ID'], config['POOL_ID'])

        # Add the tasks to the job. Pass the input files and a SAS URL
        # to the storage container for output files.
        add_tasks(batch_client, config['JOB_ID'],
                  num_of_tasks, output_container_sas_url,
                  app_id, app_version)

        # Pause execution until tasks reach Completed state.
        wait_for_tasks_to_complete(batch_client,
                                   config['JOB_ID'],
                                   datetime.timedelta(minutes=config['TIMEOUT_PERIOD']))

        print("Success! All tasks reached the 'Completed' state within the "
              "specified timeout period.")

    except batch_models.BatchErrorException as err:
        print_batch_exception(err)
        raise

    # Delete input container in storage
    # print('Deleting container [{}]...'.format(input_container_name))
    # blob_client.delete_container(input_container_name)

    # Print out some timing info
    end_time = datetime.datetime.now().replace(microsecond=0)
    print()
    print('Sample end:   {}'.format(end_time))
    print('Elapsed time: {}'.format(end_time - start_time))
    print()

    # Clean up Batch resources (if the user so chooses).
    # if query_yes_no('Delete job?') == 'yes':
    batch_client.job.delete(config['JOB_ID'])
    time.sleep(5)

    # if query_yes_no('Delete pool?') == 'yes':
    batch_client.pool.delete(config['POOL_ID'])

    # Downloading blog data
    print('Downloading blob contents...')
    blobs = blob_client.list_blobs(output_container_name)
    if not os.path.exists('results'):
        os.mkdir('results')
    for blob in blobs:
        blob_client.get_blob_to_path(output_container_name, blob.name, os.path.join('results', blob.name))

    # Delete input container in storage
    print('Deleting container [{}]...'.format(output_container_name))
    blob_client.delete_container(output_container_name)

    print("Exited normally.")
