### usage


1. Install the Required Packages
   - python version: 3.8.10
   - pip install -r requirement.txt



2. Convert ATOMIC 2020 Triples to Natural Language
   - download ATOMIC 2020 from https://allenai.org/data/atomic-2020 and put it in the directory "kb_process"
   - cd ./kb_process
   - python atomic_process.py
   - "atomic.csv" will be generated in the directory "datasets"



3. Convert ConceptNet 5.7 Triples to Natural Language
   - download conceptnet 5.7 from https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz and put it in the directory "kb_process/concept_process"
   - cd ./kb_process/concept_process
   - python extract_cpnet_relation.py
   - python conceptnet-process.py
   - "conceptnet.csv" will be generated in the directory "datasets"



4. Transferring Knowledge from Large NLI Datasets
   - download MNLI from https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce
   - download QNLI from https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0
   - download roberta-base from https://huggingface.co/roberta-base
   - download roberta-large from https://huggingface.co/roberta-large
   - put the datasets and models in the directory "/home/YOUR_USER_NAME/.cache/"
   - sh run_nli.sh  /home/YOUR_USER_NAME/.cache/
   - models roberta+QNLI/MNLI will be stored in the directory "/home/YOUR_USER_NAME/.cache/"



5. Extract Knowledge from KBs
   - run the jupyter notes in the directory "kb_extract"
   - set the value of ROOT_DIR to "/home/YOUR_USER_NAME/NLI-KB/" 
   - set the value of CACHE_DIR to "/home/YOUR_USER_NAME/.cache/"



6. Start Experiments
   - run the jupyter notes in the directory "roberta_unsup"
   - set the value of ROOT_DIR to "/home/YOUR_USER_NAME/NLI-KB/" 
   - set the value of CACHE_DIR to "/home/YOUR_USER_NAME/.cache/"

