# event_extractor

This repo includes code and models to extract event mentions from raw text, that we use in [this](https://github.com/BIU-NLP/iFACETSUM) repository.
Our code mainly relies on our [coref](https://github.com/ariecattan/coref) repo.


Run the script 

```
python predict.py --config config.json
```


Example of input document: `data/EgyptAir_Crash_docs_formatted`   
Example of output document: `data/EgyptAir_Crash_mentions`