git lfs pull --all
git lfs pull --ref origin/master

pip install git+https://github.com/keras-team/keras-tuner.git
pip install -U autokeras pandas numpy scikit-learn GPUtil humanize psutil matplotlib absl-py

# for Mac M1
#pip install -U tensorflow-metal