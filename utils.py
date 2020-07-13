import os
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

def get_workspace():
    sp_auth = ServicePrincipalAuthentication(
        tenant_id=os.environ.get('TENANT_ID'),
        service_principal_id=os.environ.get('SP_ID'),
        service_principal_password=os.environ.get('SP_PASS'),
        cloud='AzureChinaCloud'
    )

    ws = Workspace(
        subscription_id=os.environ.get('AZURE_SUB_ID'),
        resource_group=os.environ.get('AML_RG'),
        workspace_name='bba-aml-ws',
        auth=sp_auth,
        sku='enterprise'
    )
    print("Auth with AML workspace succeeded")
    return ws
