git lfs pull --all
git lfs pull --ref origin/main

pip uninstall git+https://github.com/keras-team/keras-tuner.git
pip uninstall autokeras tensorflow keras keras-tuner tensorflow-macos

pip install git+https://github.com/keras-team/keras-tuner.git
pip install -U autokeras
#pip install -U autokeras==1.1.0 tensorflow==2.15 keras==2.15.0 keras-tuner==1.4.7 autokeras pandas numpy scikit-learn GPUtil humanize psutil matplotlib absl-py pyarrow
#pip install -U autokeras pandas numpy scikit-learn GPUtil humanize psutil matplotlib absl-py pyarrow
#pip install -U autokeras==1.1.0 pandas numpy scikit-learn GPUtil humanize psutil matplotlib absl-py pyarrow
pip install -U pandas numpy scikit-learn GPUtil humanize psutil matplotlib absl-py pyarrow chardet

pip install tf-nightly keras-nightly

# for Mac M1
#pip install -U tensorflow-metal
#pip install -U tensorflow-metal==0.8.0 tensorflow-macos==2.5.0
#pip install -U tensorflow-metal==1.0.1
#pip install -U tf-nightly-macos