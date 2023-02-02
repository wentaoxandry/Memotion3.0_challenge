#!/bin/bash -u
stage=10		# start from -1 if you need to start from data download
stop_stage=100			# stage at which to stop

datasource=./Sourcedata
dsetdir=./dataset
outdir=./output
cashedir=./CASHE

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # install conda environment
    if find_in_conda_env ".*memotion.*" ; then
   	echo 'conda environment already created'
    else 
        ./install.sh
    fi
    
fi

#conda activate memotion
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    #creat json file from datasource
    mkdir -p $datasource
    cd $datasource
    git clone https://github.com/schesa/ImgFlip575K_Dataset.git
    cd ..
    python3 local/externaldataset.py ${datasource}/ImgFlip575K_Dataset $dsetdir
    python3 local/externaldataset_MMHS150k.py ${datasource}/MMHS150k $dsetdir
    python3 local/readmemotioncsv.py ${datasource}/Memotion3 $dsetdir
    python3 local/readmamicsv.py ${datasource}/MAMI $dsetdir
    python3 local/readfacebookmemes.py ${datasource}/hateful_memes $dsetdir
    jq -s 'add' $dsetdir/facebook/train.json \
    		$dsetdir/facebook/dev.json \
    		$dsetdir/external/externaltrain.json \
    		$dsetdir/MMHS150k/MMHS150k.json \
    		$dsetdir/memotion3/train.json \
    	        $dsetdir/mami/training.json  > $dsetdir/pretrain_CLIP_train.json
    jq -s 'add' $dsetdir/facebook/test.json \
    		$dsetdir/external/externaltest.json \
    	        $dsetdir/mami/test.json > $dsetdir/pretrain_CLIP_test.json
    jq -s 'add' $dsetdir/facebook/train.json \
    		$dsetdir/facebook/dev.json \
    	        $dsetdir/mami/training.json  > $dsetdir/pretrain_train.json
    jq -s 'add' $dsetdir/facebook/test.json \
    	        $dsetdir/mami/test.json > $dsetdir/pretrain_test.json
    echo 'print statistic information'
    python3 local/statistic.py ${datasource}/Memotion3 $dsetdir

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    #train text model with BERTweet
    python3 local/pretrainCLIP/pretrainCLIP_RAY.py $dsetdir ./CASHE || exit 1;
    mv RAY pretrainCLIP_image_text_match_RAY || exit 1;
    mv pretrainCLIP_image_text_match_RAY RAY_results || exit 1;

fi

###################################### Text Model ######################################################################


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    #pretrain CLIPtextmodel with external hateful dataset. Dataset is contructed for binary classification. no hate=>0 hate=>1 
    bestmodelinfo=$(python3 local/get_ray_param_model_pretrain_CLIP.py ./RAY_results/pretrainCLIP_image_text_match_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelinfo"
    #pretrain CLIPtextmodel with external hateful dataset. Dataset is contructed for binary classification. no hate=>0 hate=>1 
    python3 local/taskA/text/pretrainCLIPtext_RAY.py $dsetdir ./CASHE "$bestmodelinfo" || exit 1;
    mv RAY taskA_pretrain_text_RAY || exit 1;
    mv taskA_pretrain_text_RAY RAY_results || exit 1;
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    #use the pretrained model from stage 3 to train CLIPimagemodel with memotion dataset taskA, which has 3 classes: negative=>0 neurtal=>1 positive=>2
    bestmodelinfo=$(python3 local/get_ray_param_model_pretrain_CLIP.py ./RAY_results/taskA_pretrain_text_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelinfo"
    python3 local/taskA/text/trainCLIPtext_RAY.py $dsetdir ./CASHE "$bestmodelinfo" || exit 1;
    mv RAY taskA_train_text_RAY || exit 1;
    mv taskA_train_text_RAY RAY_results || exit 1;
    bestmodelrayinfo=$(python3 local/get_ray_param_model_taskA_2models.py ./RAY_results/taskA_train_text_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelrayinfo"
    python3 local/taskA/text/evalCLIPtext2models.py $dsetdir ${outdir}/taskA/train ./Ensemble/taskA $cashedir "$bestmodelrayinfo" || exit 1;
    python3 local/taskA/text/testCLIPtext2models.py $dsetdir ${outdir}/taskA/train ./Ensembletest/taskA $cashedir "$bestmodelrayinfo" || exit 1;
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # train for taskc but we have to train the model with different types. all 4 types.
    # pretrained model from image text match model. since taskC is not related with taskA
    # use RAY model to find the best parameters
    bestmodelinfo1=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_train_text_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelinfo1"
    for type in 'humorous' 'sarcastic' 'offensive' 'motivation'; do #'humorous' 
        python3 local/taskC/text/trainCLIPtext_RAY.py $dsetdir ./CASHE $type "$bestmodelinfo1" || exit 1;
        mv RAY1 taskC_train_text_${type}_RAY || exit 1;
        mv taskC_train_text_${type}_RAY RAY_results  || exit 1;
        bestraymodelinfo=$(python3 local/get_ray_param_model_2models.py ./RAY_results/taskC_train_text_${type}_RAY 2>&1 > /dev/null)
    	echo "$bestraymodelinfo"
        #python3 local/taskC/text/trainCLIPtext.py $dsetdir ${outdir}/taskC/train $cashedir $type  || exit 1;
        python3 local/taskC/text/evalCLIPtext_2models.py $dsetdir ${outdir}/taskC/train Ensemble/taskC $cashedir $type "$bestraymodelinfo" || exit 1;
        python3 local/taskC/text/testCLIPtext_2models.py $dsetdir ${outdir}/taskC/train ./Ensembletest/taskC $cashedir $type "$bestraymodelinfo" || exit 1;
    done   
fi

###################################### Image Model ######################################################################

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    #pretrain CLIPimagemodel with external hateful dataset. Dataset is contructed for binary classification. no hate=>0 hate=>1 
    bestmodelinfo=$(python3 local/get_ray_param_model_pretrain_CLIP.py ./RAY_results/pretrainCLIP_image_text_match_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelinfo"
    #pretrain CLIPtextmodel with external hateful dataset. Dataset is contructed for binary classification. no hate=>0 hate=>1 
    python3 local/taskA/image/pretrainCLIPimage_RAY.py $dsetdir ./CASHE "$bestmodelinfo" || exit 1;
    mv RAY taskA_pretrain_image_RAY || exit 1;
    mv taskA_pretrain_image_RAY RAY_results || exit 1;
    #bestmodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_pretrain_image_RAY 2>&1 > /dev/null) #|| exit 1;
    #echo "$bestmodelinfo"
    #python3 local/taskA/image/pretrainCLIPimage.py $dsetdir ${outdir}/taskA/pretrain $cashedir "$bestmodelinfo"|| exit 1;
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    #use the pretrained model from stage 3 to train CLIPimagemodel with memotion dataset taskA, which has 3 classes: negative=>0 neurtal=>1 positive=>2
    bestmodelinfo=$(python3 local/get_ray_param_model_pretrain_CLIP.py ./RAY_results/taskA_pretrain_image_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelinfo"
    python3 local/taskA/image/trainCLIPimage_RAY.py $dsetdir ./CASHE "$bestmodelinfo" || exit 1;
    mv RAY taskA_train_image_RAY || exit 1;
    mv taskA_train_image_RAY RAY_results || exit 1;
    bestmodelrayinfo=$(python3 local/get_ray_param_model_taskA_2models.py ./RAY_results/taskA_train_image_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelrayinfo"
    python3 local/taskA/image/evalCLIPimage_2models.py $dsetdir ${outdir}/taskA/train ./Ensemble/taskA $cashedir "$bestmodelrayinfo" || exit 1;
    python3 local/taskA/image/testCLIPimage_2models.py $dsetdir ${outdir}/taskA/train ./Ensembletest/taskA $cashedir "$bestmodelrayinfo" || exit 1;
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then

    bestmodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_train_image_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelinfo"
    
    for type in 'humorous' 'sarcastic' 'offensive' 'motivation'; do # 'humorous' 
        python3 local/taskC/image/trainCLIPimage_RAY.py $dsetdir ./CASHE $type "$bestmodelinfo" || exit 1;
        mv RAY1 taskC_train_image_${type}_RAY
        mv taskC_train_image_${type}_RAY RAY_results
        bestraymodelinfo=$(python3 local/get_ray_param_model_image_2models.py ./RAY_results/taskC_train_image_${type}_RAY 2>&1 > /dev/null)
    	echo "$bestraymodelinfo"
        #python3 local/taskC/image/trainCLIPimage.py $dsetdir ${outdir}/taskC/train $cashedir $type "$bestraymodelinfo" || exit 1;
        python3 local/taskC/image/evalCLIPimage_2models.py $dsetdir ${outdir}/taskC/train Ensemble/taskC $cashedir $type "$bestraymodelinfo"
        python3 local/taskC/image/testCLIPimage_2models.py $dsetdir ${outdir}/taskC/train Ensembletest/taskC $cashedir $type "$bestraymodelinfo"
    done  
    
fi


###################################### Multi Model ######################################################################<
<<'COMMEN'
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    #pretrain CLIPimagemodel with external hateful dataset. Dataset is contructed for binary classification. no hate=>0 hate=>1 
    #bestmodelinfo=$(python3 local/get_ray_param_model_pretrain_CLIP.py ./RAY_results/pretrainCLIP_image_text_match_RAY 2>&1 > /dev/null) #|| exit 1;
    #echo "$bestmodelinfo"
    besttextmodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_pretrain_text_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$besttextmodelinfo"
    bestimagemodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_pretrain_image_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestimagemodelinfo"
    python3 local/taskA/multi/pretrainCLIPmulti_RAY.py $dsetdir ./CASHE "$besttextmodelinfo" "$bestimagemodelinfo" || exit 1;
    mv RAY taskA_pretrain_multi_RAY
    mv taskA_pretrain_multi_RAY RAY_results
    #bestmodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_pretrain_multi_RAY 2>&1 > /dev/null)
    #python3 local/taskA/multi/pretrainCLIPmulti.py $dsetdir ${outdir}/taskA/pretrain $cashedir "$bestmodelinfo" || exit 1;
fi
COMMEN

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    #use the pretrained model from stage 3 to train CLIPimagemodel with memotion dataset taskA, which has 3 classes: negative=>0 neurtal=>1 positive=>2
    besttextmodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_train_text_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$besttextmodelinfo"
    bestimagemodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_train_image_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestimagemodelinfo"
    python3 local/taskA/multi/trainCLIPmulti_RAY.py $dsetdir ./CASHE "$besttextmodelinfo" "$bestimagemodelinfo" || exit 1; # "$bestmodelinfo" || exit 1;
    mv RAY taskA_train_multi_RAY
    mv taskA_train_multi_RAY RAY_results

    bestmodelinfo=$(python3 local/get_ray_param_model_taskA_2models.py ./RAY_results/taskA_train_multi_RAY 2>&1 > /dev/null) 
    echo "$bestmodelinfo"   
    #python3 local/taskA/multi/trainCLIPmulti.py $dsetdir ${outdir}/taskA/train $cashedir "$bestmodelinfo"  || exit 1;
    python3 local/taskA/multi/evalCLIPmulti_2models.py $dsetdir ${outdir}/taskA/train ./Ensemble/taskA $cashedir "$bestmodelinfo" || exit 1;
    python3 local/taskA/multi/testCLIPmulti_2models.py $dsetdir ${outdir}/taskA/train ./Ensembletest/taskA $cashedir "$bestmodelinfo" || exit 1;
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    bestmodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_train_multi_RAY 2>&1 > /dev/null) #|| exit 1;
    echo "$bestmodelinfo"

    for type in 'humorous' 'sarcastic' 'offensive' 'motivation'; do #'humorous' 
        besttextmodelinfo=$(python3 local/get_ray_param_model.py ./RAY_results/taskC_train_text_${type}_RAY 2>&1 > /dev/null) #|| exit 1;
        echo "$besttextmodelinfo"
        bestimagemodelinfo=$(python3 local/get_ray_param_model.py ./RAY_results/taskC_train_image_${type}_RAY 2>&1 > /dev/null) #|| exit 1;
        echo "$bestimagemodelinfo"
        python3 local/taskC/multi/trainCLIPmulti_RAY.py $dsetdir ./CASHE $type "$besttextmodelinfo" "$bestimagemodelinfo" || exit 1;
        mv RAY taskC_train_multi_${type}_RAY
        mv taskC_train_multi_${type}_RAY RAY_results
        bestraymodelinfo=$(python3 local/get_ray_param_multi_model_2models.py ./RAY_results/taskC_train_multi_${type}_RAY 2>&1 > /dev/null)
    	echo "$bestraymodelinfo"
        #python3 local/taskC/multi/trainCLIPmulti.py $dsetdir ${outdir}/taskC/train $cashedir $type "$bestraymodelinfo"  || exit 1;
        python3 local/taskC/multi/evalCLIPmulti_2models.py $dsetdir ${outdir}/taskC/train Ensemble/taskC $cashedir $type "$bestraymodelinfo"  || exit 1;
        python3 local/taskC/multi/testCLIPmulti_2models.py $dsetdir ${outdir}/taskC/train Ensembletest/taskC $cashedir $type "$bestraymodelinfo"  || exit 1;
    done

fi
<<'COMMEN'
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    # ensemble task A
    python3 local/fusiontext_image_results.py ./Ensemble/taskA/text-image-fusion ./Ensemble/taskA/text ./Ensemble/taskA/image None
    python3 local/fusiontext_image_results.py ./Ensemble/taskA/finalresults ./Ensemble/taskA/text-image-fusion ./Ensemble/taskA/multi None
    for type in 'humorous' 'sarcastic' 'offensive' 'motivation'; do
    	python3 local/fusiontext_image_results.py ./Ensemble/taskC/text-image-fusion ./Ensemble/taskC/text ./Ensemble/taskC/image $type
    	python3 local/fusiontext_image_results.py ./Ensemble/taskC/finalresults ./Ensemble/taskC/text-image-fusion ./Ensemble/taskC/multi $type
    done

fi
COMMEN
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    # try XGBooster Classifier
    bestmodelinfo=$(python3 local/get_ray_param_model_taskA.py ./RAY_results/taskA_train_multi_RAY 2>&1 > /dev/null) 
    echo "$bestmodelinfo" 
    python3 local/taskA/multi/savemultrifeatures.py $dsetdir ${outdir}/taskA/train $cashedir "$bestmodelinfo"  || exit 1;
 
    python3 local/taskA/multi/XGBooster.py $dsetdir ${outdir}/taskA/train ./CASHE
    for type in 'humorous' 'sarcastic' 'offensive' 'motivation'; do
    	bestraymodelinfo=$(python3 local/get_ray_param_multi_model.py ./RAY_results/taskC_train_multi_${type}_RAY 2>&1 > /dev/null)
    	echo "$bestraymodelinfo"
    	python3 local/taskC/multi/savemultrifeatures.py $dsetdir ${outdir}/taskC/train $cashedir $type "$bestraymodelinfo" || exit 1;
    	python3 local/taskC/multi/XGBooster.py $dsetdir ${outdir}/taskC/train $cashedir $type
    done

fi
