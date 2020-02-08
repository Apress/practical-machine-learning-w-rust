 python3 -m venv venv
 source venv/bin/activate
 pip install snips-nlu
 python -m snips_nlu download en
 snips-nlu download-all-languages
 snips-nlu train dataset.json snips.model -v
