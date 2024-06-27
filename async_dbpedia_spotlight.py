import requests
import urllib.error

import asyncio
from time import sleep
from timeit import default_timer
from concurrent.futures import ThreadPoolExecutor

import re
import spacy
import spacy_dbpedia_spotlight
import pandas as pd
import csv
import numpy as np

import argparse

OUT_CSV_DELIMITER = ','
OUT_CSV_QUOTECHAR = '"'
OUT_CSV_ESCAPECHAR = '\\'
START_TIME = default_timer()
DEFAULT_THINKTIME = 0.2
DEFAULT_CONCURRENT_WORKERS = 6
DEFAULT_CONFIDENCE = 0.6
DEFAULT_BATCH_SIZE = 1000
DBPSL_RETRIES = 3
DBPS_SLEEP_AFTER_HTTP_ERROR = 10 ## seconds

all_data = []

def pre_process(text): 
    ### FROM: https://github.com/kavgan/nlp-in-practice/tree/master/tf-idf 
    # lowercase
    text=text.lower() 
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)   
    
    # Swaps line breaks for spaces, also remove apostrophes & single quotes
    text.replace("\n", " ").replace("'"," ").replace("’"," ")
    return text

def entities_extraction(batch_num, id, text):
    try:
        doc=nlp(pre_process(str(text)))
        return (batch_num, id, [{
                "surface_form": ent.text,
                "url": ent.kb_id_,
                "types": ent._.dbpedia_raw_result['@types'].split(",")} for ent in doc.ents],
                None,None)
                #doc.user_data["status"],doc.user_data["error_message"]) ## TODO: extract status and error_message durinig entity linking in file spacy_dbpedia_spotlight/entity_linker.py 
    except TypeError as e:
        return (batch_num, id,[str(e)])
    except urllib.error.HTTPError as e:
        return (batch_num, id,[str(e)])


def spotlight_old(row):
    return entities_extraction(row['id'],row['summary'])

def spotlight(csv_writer, batch_num, id,text, flush=False, think_time = DEFAULT_THINKTIME):
    sleep(think_time)
    success = False
    for i in range (0,DBPSL_RETRIES):
        try:
            data = entities_extraction(batch_num, id, text)
        except Exception as e:
            print("{0:<30} {1:>20}".format(id, "Retry after exception calling SBpedia Spotlight:", e),flush=flush)
            sleep(DBPS_SLEEP_AFTER_HTTP_ERROR)
        else:
            success = True
            break

    if success:
        elapsed_time = default_timer() - START_TIME
        completed_at = "{:5.2f}s".format(elapsed_time)
        print("{0:<30} {1:>20}".format(id, completed_at),flush=flush)
        try:
            csv_writer.writerow(data)
            print("{0:<30} {1:>20}".format(id, "written"),flush=flush)
        except Exception as e:
            print("{0:<30} {1:>20}".format(id, "exception writing line to csv"),flush=flush)
            print("{0:<30} {1:>20}".format(id, str(data)),flush=flush)
            print("exception: ",e)
        else:
            return data
    else:
        print("{0:<30} {1:>20}".format(id, "skipping... exception calling SBpedia Spotlight"),flush=flush)
        print("{0:<30} {1:>20}".format(id, "Skipping after %s retries" % DBPSL_RETRIES),flush=flush)
        



async def start_async_process(df, batch_num, csv_output  = 'output.csv', text_column = 'text', id_column = 'id', max_workers=16,think_time = DEFAULT_THINKTIME):
    bufsize = 2 * max_workers # we write to csv in bunch of 2 * max_workers lines
    print("{0:<30} {1:>20}".format("No", "Completed at"))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with open(csv_output, 'a', bufsize) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=OUT_CSV_DELIMITER, quotechar=OUT_CSV_QUOTECHAR, escapechar=OUT_CSV_ESCAPECHAR, quoting=csv.QUOTE_ALL)
            csv_writer.writerow(['batch_num', 'id', 'entities', 'status', 'error'])
            loop = asyncio.get_event_loop()
            START_TIME = default_timer()
            tasks = [
                loop.run_in_executor(
                    executor,
                    spotlight,
                    *(csv_writer, batch_num, row[id_column], row[text_column], True, think_time)
                )
                for index, row in df.iterrows()
            ]
            for response in await asyncio.gather(*tasks):
                pass



                



nlp = spacy.load('en_core_web_sm')

tourism_types = "DBpedia:Activity,DBpedia:Food,DBpedia:Holiday,DBpedia:MeanOfTransportation,DBpedia:Place,Schema:Event,Schema:Place"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_input", help="CSV input filename with id and text data to apply entity linking")
    parser.add_argument("csv_output", help="CSV output filename with entities found")
    
    parser.add_argument("--text_column", help="Name of the csv column with the text to be analyzed", default = 'text')
    parser.add_argument("--id_column", help="Name of the csv column with the unique id for the text", default = 'id')
    parser.add_argument("--max_workers", help="Number of concurrent workers used to call DBpedia Spotlight", default = DEFAULT_CONCURRENT_WORKERS, type=int)
    parser.add_argument("--batch_size", help="Number of lines for a batch of texts from csv to process with DBpedia Spotlight", default = DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("--batch_skip", help="Number of batches to skip from start", default = 0, type=int)
    parser.add_argument("--num_records", help="Number of records to process. Default is to take all records.", default = None, type=int)
    parser.add_argument("--skip_until_id", help="Skip csv lines until the id is found.", default = None, type=str)
    parser.add_argument("--think_time", help="Number of seconds (also fractions) to wait before executing a call to DBpedia Spotlight", default = DEFAULT_THINKTIME, type=float)
    parser.add_argument("--confidence", help="Confidence threshold used used to call DBpedia Spotlight", default = DEFAULT_CONFIDENCE, type=float)
    parser.add_argument("--types", help="DBpedia classes to fileter e.g. DBpedia:Activity,DBpedia:Food, if special keyword 'tourism' is used the tourism filter is applied (DBpedia:Activity,DBpedia:Food,DBpedia:Holiday,DBpedia:MeanOfTransportation,DBpedia:Place,Schema:Event,Schema:Place),  if not specified the default behaviour is to get everything.", default = None, type=str)

    

    args = parser.parse_args()
    
    parameters = {
        "confidence": args.confidence,
        "types": args.types
    }

    if args.types == 'tourism':
        parameters["types"] =  tourism_types
    elif args.types is not None:
        parameters["types"] =  args.types
   


    print("DBpedia Spotlight used parameters:", parameters)
    nlp.add_pipe('dbpedia_spotlight',config=parameters)

    total_df = pd.read_csv(args.csv_input, quoting=csv.QUOTE_MINIMAL, encoding='utf-8',on_bad_lines='warn')
    if args.skip_until_id is None:
        start = 0
    else:
        indices = total_df.index[total_df[args.id_column].astype('str') == args.skip_until_id].tolist() ## find indices where the id is equal to the skip
        try:
            start = indices[0] + 1 ## take first index of the list 
        except IndexError:
            start = 0 ## if the id was not found the index list il empty so we start from the beginning of the data frame
    
    if args.num_records is None:
        end = -1
    elif args.num_records > 0:
        end = start + args.num_records
    else:
        end = -1
    
    print("Slicing csv rows. Start: %s, End: %s" % (start, end))
    selected_df = total_df.iloc[start:end]

    num_lines = selected_df.shape[0]
    print("num lines: ", num_lines)

    num_steps = (num_lines + args.batch_size - 1) // args.batch_size

    print("num lines: %s, batch size: %s, num steps: %s" % (num_lines, args.batch_size, num_steps) )

    list_df = np.array_split(selected_df, num_steps)
    for batch_num, df in enumerate(list_df[args.batch_skip:]):
        print("Batch num: %s #########################" % (batch_num + args.batch_skip))
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(start_async_process(df, batch_num, csv_output = args.csv_output, text_column = args.text_column, id_column = args.id_column, max_workers = args.max_workers, think_time = args.think_time))
        loop.run_until_complete(future)
        sleep(5)