from azureml.core.conda_dependencies import CondaDependencies

cd = CondaDependencies.create()
cd.add_tensorflow_conda_package()
cd.add_conda_package('keras<=2.3.1')
cd.add_pip_package("azureml-defaults")
cd.save_to_file(base_directory='./', conda_file_path='env.yml')

print(cd.serialize_to_string())