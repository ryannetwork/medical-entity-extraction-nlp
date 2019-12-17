from preprocessor import PreProcessor
from MIMIC_Bert_NER import *



if __name__ == '__main__':
    #Training data
    save_path = "./cleaned_files_train"
    concept_path = "../dnc_code/medical_data/train_data/concept"
    text_path = "../dnc_code/medical_data/train_data/txt"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    preproc = PreProcessor(concept_path,text_path,save_path)

    preproc.pre_process()

    #Train BERT
    bert = MIMICBertNER(save_path)
    bert.get_inputs()

    #bert_out_address = './saved_models/bert_out_model/mimic_bert'

    model = bert.train()

    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(bert.bert_out_address, "pytorch_model.bin")
    output_config_file = os.path.join(bert.bert_out_address, "config.json")

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    bert.tokenizer.save_vocabulary(bert.bert_out_address)

    #Validation data metrics
    bert.evaluate()

    #Test data
    save_path = "./cleaned_files_test"
    concept_path = "../dnc_code/medical_data/test_data/concept"
    text_path = "../dnc_code/medical_data/test_data/txt"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    preproc = PreProcessor(concept_path,text_path,save_path)
    preproc.pre_process()

    #Test BERT
    bert_test = MIMICBertNER(saved_path)
    bert_test.get_inputs("test")
    bert_test.evaluate()
