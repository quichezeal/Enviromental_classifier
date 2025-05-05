Environmental Research Text Classifier

A text-based classifier to identify environmental research-related work and assign pre-defined subclasses and keywords to the works identified.

Prerequisites

- Python 3.10.11
- pip

Required files

- sys_stopwords.txt
- external_work_combined2.csv (Or alternative .csv file containing the metadata to train on. Must have the following columns: Item Title, Abstract, URL, Subcategories. Example file: https://drive.google.com/file/d/12dU2GNKfxKvk0jkWxKBcfuAcRyJpUwWB/view?usp=sharing)
- R3 metadata.csv (if multiple .csv files exist, follow Env_control.py in-script prompt or run combine_sheets.py to merge them into one single file)

Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/quichezeal/Enviromental_classifier.git
   cd Enviromental_classifier

2. Single script run:
   run Env_control.py
