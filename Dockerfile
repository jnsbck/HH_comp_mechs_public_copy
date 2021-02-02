FROM python:3.7.9

WORKDIR /usr/src/app

# install python dependencies
COPY docker_dependencies.txt ./
RUN pip install --no-cache-dir -r docker_dependencies.txt

# copy dependencies
COPY ./code/ephys_extractor.py ./code/
COPY ./code/ephys_features.py ./code/
COPY ./code/EphysDataHelper ./code/EphysDataHelper

# copy trained density estimators
COPY ./code/trained_posteriors ./code/trained_posteriors

# compile C++ HH code into shared lib
RUN g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./code/EphysDataHelper/simulators/HHsimulatorWrapper.c -o ./code/EphysDataHelper/simulators/HHsimulatorWrapper`python3-config --extension-suffix`

# copy working data and working nb
COPY ./code/Finding_Compensatory_Mechanisms_in_HH.ipynb ./code/
COPY ./data/04_03_2020_sample_4.mat ./data/

# change workdir back to code dir
WORKDIR /usr/src/app/code

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
