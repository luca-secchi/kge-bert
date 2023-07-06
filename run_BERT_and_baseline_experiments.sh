#!/bin/bashs
##### GENERATE KGE-BERT-full, KGE-BERT-num and KGE-BERT-1hot RESULTS IN THE PAPER


BERT_MODELS_DIR=pre-trained_models
OUTPUT_DIR=./output_results/

EPOCHS=5 ## number of epochs to train
MAX_EPOCH_REPEAT=0 ## number of time we can repeat an epoch if the avg loss is not decreasing
BATCH_SIZE=8 ## batch size for training
DROPOUT=0.1 ## dropout for training
SEQ_LENGTH=200 ## mak num tokens in sentence (max 512)
TIMESTAMP=`date +"%Y%m%d_%H%M"`
DATETIME=`date +"%Y-%m-%dT%H:%M"`
export BERT_MODELS_DIR

########################## METADATA TYPE SETTINGS #####################################
#META_TYPE=""
META_TYPE="all_meta_norm_" ## if we want to use data files with all metadata normalized
USE_VALID_META="False" ## leave it to False
#######################################################################################

## Labels for experiments
# "airbnb__bert_ext_all_meta_norm_hot_encoding": "KGE-BERT-injected-full",
# "airbnb__bert_all_meta_norm_hot_encoding": "KGE-BERT-full",
# "airbnb__bert_all_meta_norm": "KGE-BERT-num",
# "airbnb__bert_hot_encoding": "KGE-BERT-1hot",
# "airbnb__bert": "BERT",
# "airbnb__bert_ext_v2": "BERT-injected",
# "airbnb__linear_all_meta_hot_encoding": "LOGISTIC REGRESSION",
EXPERIMENTS=("airbnb__bert" "airbnb__linear_all_meta_hot_encoding")


DATA_DIR="./airbnb_bert_experiments/bert_input_data"

## Labels for classification tasks
# "avail365": "Relevance classification task",
# "num_reviews": "User interest classification task",
# "price": "Price value classification task",
# "rev_score": "User appreciation classification task"
TASKS=("avail365" "num_reviews" "price" "rev_score") 

DATASET="airbnb_london_20220910"
NUM_RUN=1
FIRST_RUN=1

### If LOOKUP_NAME is not empty we use always the same lookup file for all targets, otherwise a different lookup file is used for each target using LOOKUP_SUFFIX
LOOKUP_NAME="all_he_listing_types-amenities-tao__dbpedia_airbnb_london_20220910_id2embedding.pickle"
VECTORTYPE="hot_encoding_tao_dbp"

#### Use "_extended_tao_meta_lt_v2" if you want to use the listing description extended with the list of amenities offered and numerical features
EXTENDED_DESCRIPTION=""

RANDOM_SEED_BASE=200

## If SEQ_LENGTH is not set we use 512 max for Bert
if [ -z ${SEQ_LENGTH+x} ]; then SEQ_LENGTH=512; fi
if [ -z ${USE_VALID_META+x} ]; then USE_VALID_META="False"; fi

### dev and test sizes are always the same so that we can comapre results with different training sizes
MAX_DEV_SIZE=1800 ## max size for all targets
MAX_TEST_SIZE=1080

CACHE_ENABLED="False" ## speed-up the training only for KGE-BERT experiments
## first we execute all experiments for all targets with a specific train size
## then we change the train size and repeat all experiments for all targets
## then we repeat for the desired number of times

for MAX_TRAIN_SIZE in `seq 3000 3000 12000`;do
  for i in `seq ${FIRST_RUN} ${NUM_RUN}`;do
    RANDOM_SEED=$((RANDOM_SEED_BASE * i)) 
    EMBEDDING_CACHE_FILE_NAME="embedding_cache_rand_$RANDOM_SEED.pickle" ### Saved under extras directory

    for EXP_NAME in ${EXPERIMENTS[@]};do
      rm -rf $OUTPUT_DIR/${EXP_NAME}/
      TASK_EXECUTION_DIR=${EXP_NAME}__train-${MAX_TRAIN_SIZE}__execution-${TIMESTAMP}
      mkdir $OUTPUT_DIR/$TASK_EXECUTION_DIR

      for t in ${TASKS[@]};do 
	mkdir $OUTPUT_DIR/${EXP_NAME}/

	### If LOOKUP_NAME is not empty we use always the same lookup file for all targets, otherwise a different lookup file is used for each target using LOOKUP_SUFFIX
	if [ -z ${LOOKUP_NAME+x} ]; then 
        	LOOKUP_NAME=${t}$LOOKUP_SUFFIX 
	fi

        EXPERIMENT_DIR=${EXP_NAME}_${t}_${DATASET}_${TIMESTAMP}__trainsize-${MAX_TRAIN_SIZE}__epocs-${EPOCHS}__dropout-${DROPOUT}__run-${i}
        RUN_METADATA="\"time\":\"${DATETIME}\", \"target\": \"${t}\", \"experiment\":\"${EXP_NAME}\",\"epochs\":${EPOCHS},\"max_train_size\":\"${MAX_TRAIN_SIZE}\",\"vector_type\":\"${VECTORTYPE}\",\"tot_runs\":${NUM_RUN},\"run\":${i},\"random_seed\": \"${RANDOM_SEED}\", \"kg\": \"n/a\",\"dropout\": \"${DROPOUT}\",\"batch_size\": \"${BATCH_SIZE}\",\"text_extension\": \"${EXTENDED_DESCRIPTION}\",\"cached\": \"${CACHE_ENABLED}\",\"path\":\"$OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR\""
        #RUN_METADATA="\"time\":\"${DATETIME}\", \"target\": \"${t}\", \"experiment\":\"${EXP_NAME}\",\"epochs\":${EPOCHS},\"max_train_size\":\"${MAX_TRAIN_SIZE}\",\"vector_type\":\"n/a\",          \"tot_runs\":${NUM_RUN},\"run\":${i}, \"random_seed\": \"${RANDOM_SEED}\", \"kg\": \"${KG_NAME}\",\"path\":\"$OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR\""

	OUTPU_LOG_FILE=$OUTPUT_DIR/${EXP_NAME}/execution.log
	echo "######################################################### NEW EXPERIMENT RUN #################" | tee -a $OUTPU_LOG_FILE 
        echo "Random seed: "$RANDOM_SEED
        echo "Cache file: "$EMBEDDING_CACHE_FILE_NAME
        echo $RUN_METADATA 
	if [ -z ${META_TYPE} ]; then 
		echo "Using a subset of AirBnb metadata." | tee -a $OUTPU_LOG_FILE 
	else
		if [ "$USE_VALID_META" == "True" ]; then
			echo "Using valid AirBnb metadata." | tee -a $OUTPU_LOG_FILE
		else	
			echo "Using all AirBnb metadata." | tee -a $OUTPU_LOG_FILE
		fi
	fi


        echo "Lookup file: $LOOKUP_NAME" | tee -a $OUTPU_LOG_FILE
        echo "Vectory type: $VECTORTYPE" | tee -a $OUTPU_LOG_FILE
	echo "Working on dir: $OUTPUT_DIR/${EXP_NAME}" | tee -a $OUTPU_LOG_FILE
	echo "Using files:" | tee -a $OUTPU_LOG_FILE
	echo ${DATA_DIR}/train_${META_TYPE}${t}${EXTENDED_DESCRIPTION}_${DATASET}.pickle  | tee -a $OUTPU_LOG_FILE
	echo ${DATA_DIR}/dev_${META_TYPE}${t}${EXTENDED_DESCRIPTION}_${DATASET}.pickle  | tee -a $OUTPU_LOG_FILE
	echo ${DATA_DIR}/test_${META_TYPE}${t}${EXTENDED_DESCRIPTION}_${DATASET}.pickle | tee -a $OUTPU_LOG_FILE
        #debugpy-run cli.py \
	start=`date +%s`
       	python -u cli.py \
              run_on_val_and_test \
              ${EXP_NAME} \
              0 \
              ./extras/ \
              ${DATA_DIR}/train_${META_TYPE}${t}${EXTENDED_DESCRIPTION}_${DATASET}.pickle \
              ${DATA_DIR}/dev_${META_TYPE}${t}${EXTENDED_DESCRIPTION}_${DATASET}.pickle \
              ${DATA_DIR}/test_${META_TYPE}${t}${EXTENDED_DESCRIPTION}_${DATASET}.pickle \
              $OUTPUT_DIR \
              --epochs $EPOCHS --random_state ${RANDOM_SEED} \
	      --dropout $DROPOUT \
	      --max_repeat $MAX_EPOCH_REPEAT \
              --batch_size $BATCH_SIZE --seq_length ${SEQ_LENGTH} \
              --max_training_size $MAX_TRAIN_SIZE \
              --max_dev_size $MAX_DEV_SIZE \
              --max_test_size $MAX_TEST_SIZE \
              --lookup_name $LOOKUP_NAME \
	      --evaluate_train True \
	      --use_valid_meta $USE_VALID_META \
	      --enable_cache $CACHE_ENABLED --update_embedding_cache False \
	      | tee -a $OUTPU_LOG_FILE

        end=`date +%s`
	elapsed=$(expr $end - $start)
	echo "------------------------------------------------------------" | tee -a $OUTPU_LOG_FILE
	echo "Execution time was $elapsed seconds" | tee -a $OUTPU_LOG_FILE

	TOT_EPOCHS=$(cat $OUTPUT_DIR/${EXP_NAME}/execution.log | sed -n 's/.*__train_epochs:\(.*\)__.*/\1/p') 
        echo "TOT: $TOT_EPOCHS" | tee -a $OUTPU_LOG_FILE

	AVG_LOSS_ALWAYS_DECREASE=$(cat $OUTPUT_DIR/${EXP_NAME}/execution.log | sed -n 's/.*__avg_loss_always_decreasing:\(.*\)__.*/\1/p') 
	echo "AVG_LOSS_ALWAYS_DECREASE: $AVG_LOSS_ALWAYS_DECREASE" | tee -a $OUTPU_LOG_FILE
	echo "------------------------------------------------------------" | tee -a $OUTPU_LOG_FILE

	## Update metadata with num epocs and loss decrease flag
	RUN_METADATA="\"time\":\"${DATETIME}\", \"target\": \"${t}\", \"experiment\":\"${EXP_NAME}\",\"epochs\":${EPOCHS},\"max_train_size\":\"${MAX_TRAIN_SIZE}\",\"vector_type\":\"${VECTORTYPE}\",\"tot_runs\":${NUM_RUN},\"run\":${i},\"random_seed\": \"${RANDOM_SEED}\", \"kg\": \"n/a\",\"tot_epoch\":\"$TOT_EPOCHS\",\"loss_always_decrease\":\"$AVG_LOSS_ALWAYS_DECREASE\",\"execution_time_sec\":\"$elapsed\",\"dropout\": \"${DROPOUT}\",\"batch_size\": \"${BATCH_SIZE}\",\"text_extension\": \"${EXTENDED_DESCRIPTION}\",\"cached\": \"${CACHE_ENABLED}\",\"path\":\"$OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR\""
        echo $RUN_METADATA  | tee -a $OUTPU_LOG_FILE
	
	if test -f "$OUTPUT_DIR/${EXP_NAME}/train_report.txt"; then
	   echo "Training report:" | tee -a $OUTPU_LOG_FILE
	   cat $OUTPUT_DIR/${EXP_NAME}/train_report.txt | tee -a $OUTPU_LOG_FILE
	fi
		
	echo "Validation report:" | tee -a $OUTPU_LOG_FILE
	cat $OUTPUT_DIR/${EXP_NAME}/val_report.txt | tee -a $OUTPU_LOG_FILE

	echo "Test report:" | tee -a $OUTPU_LOG_FILE
        cat $OUTPUT_DIR/${EXP_NAME}/test_report.txt | tee -a $OUTPU_LOG_FILE
        
        mv $OUTPUT_DIR/${EXP_NAME} $OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR
        echo "{ ${RUN_METADATA} }" > $OUTPUT_DIR/$TASK_EXECUTION_DIR/$EXPERIMENT_DIR/run_metadata.json 
      
      done
    done
  done
done
