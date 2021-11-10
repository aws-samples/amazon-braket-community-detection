# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import shutil
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

def download_graphs(graph_url, data_dir = "./graph_data"):
    """
    Download graph .zip files from web URL
    
    :param graph_url: dict, with a format of {'graph_name': 'url'}
    :param data_dir: str, the directory path to store graph data
    """
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created ./graph_data directory in local machine to store graph data.")
    
    for graph_name in graph_url.keys():
        url = graph_url[graph_name]
        with urlopen(url) as zr:
            with ZipFile(BytesIO(zr.read())) as zf:
                zf.extractall(data_dir)

def clean_graph_data(graph_files, data_dir = "./graph_data"):
    """
    Clean graph data by removing header lines
    
    :param graph_files: dict, with a format of {'graph_name': {'file': str, 'lines_to_skip': int}}
    :param data_dir: str, the directory path to graph data
    """
    
    for graph_name in graph_files.keys():
        
        # create a subfolder for each graph and save its file with header lines removed
        graph_folder = os.path.join(data_dir, graph_name)
        if not os.path.exists(graph_folder):
            os.makedirs(graph_folder)

        raw_file = os.path.join(data_dir, graph_files[graph_name]['file'])
        new_file = os.path.join(graph_folder, graph_files[graph_name]['file'])

        with open(raw_file, 'r') as f_raw:
            data = f_raw.read().splitlines(True) 
        with open(new_file, 'w') as f_new:
            f_new.writelines(data[graph_files[graph_name]['lines_to_skip']:])