#!/usr/bin/env python3
from __future__ import print_function
import os
import sys
import logging
import yaml
from natsort import natsorted
from zipfile import ZipFile
from time import sleep
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

import azure.storage.blob as azure_blob
import azure.mgmt.batch as batch_management
import azure.mgmt.batch.models as batch_mgmt_models
import azure.identity as identity

BLOB_URL_TEMPLATE = 'https://{}.blob.core.windows.net/{}/{}'

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
    logging.debug(natsorted(versions))
    return natsorted(versions)[-1]

def zip_and_upload(path_to_binary, app_id, app_version, app_package):
    # Zip file
    abs_path_to_binary = os.path.abspath(os.path.expanduser(path_to_binary))
    binary_dir = os.path.dirname(abs_path_to_binary)
    file = os.path.basename(path_to_binary)
    file_name = os.path.splitext(file)[0]
    zipped_file = file_name + '.zip'
    path_to_zipped_app = os.path.join(binary_dir, zipped_file)
    with ZipFile(path_to_zipped_app, 'w') as zip_f:
        zip_f.write(abs_path_to_binary, file)

    blob_client = azure_blob.BlockBlobService(
        config['STORAGE_ACCOUNT_NAME'],
        config['STORAGE_ACCOUNT_KEY']
    )

    # Create the blob
    container_name = app_package.storage_url.split('/')[3]
    #etag = app_package.etag.split('/')[1][1:-1]
    blob_client.create_container(container_name, fail_on_exist=False)
    while not blob_client.exists(container_name):
        sleep(1)
    blob_properties = blob_client.create_blob_from_path(
        container_name=container_name,
        blob_name=app_version,
        file_path=path_to_zipped_app
    )
    return blob_properties.etag

def create_application_package(batch_mgmt_client, app_id, app_version):
    app_object = batch_mgmt_client.application_package.create(
        config['RESOURCE_GROUP_NAME'],
        config['BATCH_ACCOUNT_NAME'],
        app_id,
        app_version,
    )
    return app_object

def activate_application_package(batch_mgmt_client, app_id, app_version):
    batch_mgmt_client.application_package.activate(
        config['RESOURCE_GROUP_NAME'],
        config['BATCH_ACCOUNT_NAME'],
        app_id,
        app_version,
        parameters = batch_mgmt_models.ActivateApplicationPackageParameters(
            format='zip')
    )

def set_application_as_default(batch_mgmt_client, app_id, app_version):
    batch_mgmt_client.application.update(
        config['RESOURCE_GROUP_NAME'],
        config['BATCH_ACCOUNT_NAME'],
        app_id,
        parameters = {
            'default_version': app_version,
            'allow_updates': True,
            'display_name': app_version
        }
    )

def increase_version(version):
    if len(version) == 0:
        new_version = '1.0.0'
    major, minor, _ = version.split('.')
    if int(minor) + 1 == 10:
        new_minor = '0'
        new_major = str(int(major) + 1)
    else:
        new_minor = str(int(minor) + 1)
        new_major = major
    new_version = '.'.join([new_major, new_minor, '0'])
    return new_version

if __name__ == "__main__":
    # Check if application already exists
    print("Checking if application binary exists... ", end='')
    binary_file_found = False
    for folder in FOLDERS:
        path_to_binary_file = os.path.abspath(os.path.join(folder, config['APP_BINARY']))
        if os.path.exists(path_to_binary_file):
            binary_file_found = True
            break
    if not binary_file_found:
        print('Application binary not found.')
        exit()
    print(" done.")

    app_id = config['APP_ID']
    print("Authenticating and retrieving latest version of {}... ".format(app_id), end='')
    # Log in
    credential = identity.AzureCliCredential()
    # Create batch management client
    batch_mgmt_client = batch_management.BatchManagementClient(
        credential = credential, subscription_id=config['SUBSCRIPTION_ID'])
    version = get_latest_app_version(batch_mgmt_client, app_id)
    print(" latest version: {}.".format(version))
    new_version = increase_version(version)
    print("Creating package for {} version {}... ".format(app_id, new_version), end='')
    app_package = create_application_package(batch_mgmt_client, app_id, new_version)
    print(" done.")
    print("Zipping and uploading application to Azure blob...", end='')
    sys.stdout.flush()
    zip_and_upload(path_to_binary_file, app_id, new_version, app_package)
    print(" done.")
    print("Activating application and setting it as default ...", end='')
    activate_application_package(batch_mgmt_client, app_id, new_version)
    set_application_as_default(batch_mgmt_client, app_id, new_version)
    print(" done.")
    