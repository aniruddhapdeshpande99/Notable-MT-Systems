# Notable-MT-Systems
By : Aniruddha Deshpande (20161058)

# Directory Structure
.
├── Data
│   └── enghin
│       ├── dev.en
│       ├── dev.hi
│       ├── test.en
│       ├── test.hi
│       ├── train.en
│       └── train.hi
├── Models
│   ├── Attn_Seq2Seq_NMT
│   │   ├── Decoder
│   │   │   └── eng_to_hin_decoder.pth
│   │   └── Encoder
│   │       └── eng_to_hin_encoder.pth
│   ├── Eff_Attn_NMT
│   │   ├── Attention
│   │   │   ├── eng_to_hin_attention_params_concat
│   │   │   ├── eng_to_hin_attention_params_dot
│   │   │   └── eng_to_hin_attention_params_general
│   │   ├── Decoder
│   │   │   ├── eng_to_hin_decoder_params_concat
│   │   │   ├── eng_to_hin_decoder_params_dot
│   │   │   └── eng_to_hin_decoder_params_general
│   │   └── Encoder
│   │       ├── eng_to_hin_encoder_params_concat
│   │       ├── eng_to_hin_encoder_params_dot
│   │       └── eng_to_hin_encoder_params_general
│   └── Seq2Seq_NMT
│       ├── Decoder
│       │   └── eng_to_hin_decoder.pth
│       └── Encoder
│           └── eng_to_hin_encoder.pth
├── Question
│   └── Assignment 4
└── Source
    ├── Attn_Seq2Seq_NMT
    │   └── attn_seq2seq_nmt.ipynb
    ├── Eff_Attn_NMT
    │   └── eff_attn_nmt.ipynb
    └── Seq2Seq_NMT
  	└── seq2seq_nmt.ipynb

# Drive Link to Code and Model
1. Please refer to this [drive link](https://drive.google.com/drive/folders/19YhLaLd6Tg2U5pGlAiu7jPV1GlIKI0Ws?usp=sharing) to access the Code and the Model.
2. The Code can be found in the `Source` Folder:
- Sequence to Sequence Learning with Neural Networks : In Folder `Seq2Seq_NMT`
- Neural Machine Translation By Jointly Learning To Align And Translate : In Folder `Attn_Seq2Seq_NMT`
- Effective Approaches to Attention-based Neural Machine Translation : In Folder `Eff_Attn_NMT`
3. The code is written in IPython Notebook and the **References** used for the code are linked in the notebook.
4. The Models can be found in the `Models` Folder:
- Sequence to Sequence Learning with Neural Networks : In Folder `Seq2Seq_NMT` (In Folders `Encoder` and `Decoder`)
- Neural Machine Translation By Jointly Learning To Align And Translate : In Folder `Attn_Seq2Seq_NMT` (In Folders `Encoder` and `Decoder`)
- Effective Approaches to Attention-based Neural Machine Translation : In Folder `Eff_Attn_NMT` (In Folders `Encoder` and `Decoder`)
5. Sample Outputs can be seen within the Notebook as well. (can be randomly generated using `evaluateRandomly()` function)

# Prerequisites
1. torch (1.0.1.post2)
2. torchvision (0.2.1)
3. jupyter (1.0.0)

# Analysis 
- Note : English-Hindi Language pair is chosen in this case and the models are trained for English to Hindi Translation.

## Sequence to Sequence Learning with Neural Networks
1. Sample Sentence Translations :
English Sentence : can be reached here in three hours from pathankot .
Actual Translation : पठानकोट से यहाँ तीन घंटे में पहुँचा जा सकता है ।
Model's Translation : यहाँ से तीन से से से सकता में सकता है <EOS>

English Sentence : any one technique could be selected for massaging .
Actual Translation : मालिश के लिए कोई भी एक नुस्खा चुना जा सकता है ।
Model's Translation : एक एक एक लिए एक के लिए <EOS>

English Sentence : the cough can be dry and phlegmatic as well .
Actual Translation : खाँसी सूखी भी हो सकती है और बलगम वाली भी ।
Model's Translation : और के भी भी हो । । <EOS>

English Sentence : first that in which some part of the breast only is removed as per need .
Actual Translation : पहली वह जिसमें केवल ब्रेस्ट का आवश्यकतानुसार थोड़ा हिस्सा ही निकाला जाता है ।
Model's Translation : कुछ कुछ के है है के है में ही है है है में ही है में है है में । है । <EOS>

English Sentence : slowly try to bring the hand and legs together which will make the shape of the body like a chakra .
Actual Translation : धीरे-धीरे हाथ एवं पैरों को समीप लाने का प्रयत्‍न करें , जिससे शरीर की चक्र जैसी आकृति बन जाये ।
Model's Translation : शरीर को शरीर को , और , , , कर <EOS>

English Sentence : now massaging on your head leave for ten to fifteen minutes .
Actual Translation : अब सिर पर मसाज करते हुए दस से पन्द्रह मिनट तक छोड़ दें ।
Model's Translation : अब पर तक मिनट पर मिनट तक मिनट । <EOS>

English Sentence : the vaccine of chicken pox is given after the age of one year .
Actual Translation : चिकन पॉक्‍स का टीका एक वर्ष की आयु के बाद लगाया जाता है ।
Model's Translation : एक की बाद के बाद आयु बाद की आयु के बाद है है <EOS>

English Sentence : saving a mental patient from falling ill again .
Actual Translation : किसी मानसिक रोगी को दुबारा रोगी होने से बचाना ।
Model's Translation : रोगी रोगी रोगी को रोगी को रोगी है । । <EOS>

English Sentence : in hastinapur bhagwan made big kings follower of the jain religion with his preachings .
Actual Translation : हस्तिनापुर में भगवान ने अपने प्रवचनों से 6 बड़े राजाओं को जैन धर्म का अनुयायी बनाया ।
Model's Translation : इस में में के में के में में , की हुए में की में । में । <EOS>

2. Note that above translations are at a high average loss of 4.8061 post 3 Hrs of training on GTX 1050.
3. Repeatitions are observed heavily due to lack of coverage.
4. Completely misses out on translation of few kinds of words (See below as to how they are generated by the Attention based model.

## Neural Machine Translation By Jointly Learning To Align And Translate

1. Sample Sentence Translations :
English Sentence : can be reached here in three hours from pathankot .
Actual Translation : पठानकोट से यहाँ तीन घंटे में पहुँचा जा सकता है ।
Model's Translation : यहाँ तीन घंटे से तीन घंटे । <EOS>

English Sentence : any one technique could be selected for massaging .
Actual Translation : मालिश के लिए कोई भी एक नुस्खा चुना जा सकता है ।
Model's Translation : एक भी भी भी है के है है है । है । <EOS>

English Sentence : this is very near the border of china in north sikkim .
Actual Translation : यह उत्तर सिक्किम में चीन की सीमा के बहुत नजदीक है ।
Model's Translation : इस के यह में के के के में के है । <EOS>

English Sentence : the cough can be dry and phlegmatic as well .
Actual Translation : खाँसी सूखी भी हो सकती है और बलगम वाली भी ।
Model's Translation : खाँसी और भी भी भी भी है । । है । <EOS>

English Sentence : first that in which some part of the breast only is removed as per need .
Actual Translation : पहली वह जिसमें केवल ब्रेस्ट का आवश्यकतानुसार थोड़ा हिस्सा ही निकाला जाता है ।
Model's Translation : कुछ ही जो है है में है है जो में है में है में है है में है । । । है । <EOS>

English Sentence : slowly try to bring the hand and legs together which will make the shape of the body like a chakra .
Actual Translation : धीरे-धीरे हाथ एवं पैरों को समीप लाने का प्रयत्‍न करें , जिससे शरीर की चक्र जैसी आकृति बन जाये ।
Model's Translation : शरीर को के के के के के , के के है के के , के । है । <EOS>

English Sentence : now massaging on your head leave for ten to fifteen minutes .
Actual Translation : अब सिर पर मसाज करते हुए दस से पन्द्रह मिनट तक छोड़ दें ।
Model's Translation : अब आपके लिए मिनट मिनट को की मिनट । । । । <EOS>

English Sentence : the vaccine of chicken pox is given after the age of one year .
Actual Translation : चिकन पॉक्‍स का टीका एक वर्ष की आयु के बाद लगाया जाता है ।
Model's Translation : वर्ष के के वर्ष के है के के । है । । <EOS>

English Sentence : saving a mental patient from falling ill again .
Actual Translation : किसी मानसिक रोगी को दुबारा रोगी होने से बचाना ।
Model's Translation : रोगी रोगी के के रोगी के रोगी है । । । । । । <EOS>

English Sentence : in hastinapur bhagwan made big kings follower of the jain religion with his preachings .
Actual Translation : हस्तिनापुर में भगवान ने अपने प्रवचनों से 6 बड़े राजाओं को जैन धर्म का अनुयायी बनाया ।
Model's Translation : जैन धर्म ने अपने में में के में में के में में की में की <EOS>

2. Note that above translations are at a high average loss of 4.8087 post 3 Hrs of training on GTX 1050.
3. Repeatitions are observed heavily (due to lack of coverage) and higher attention can be observed to the final sections of the sentence (at this stage of training) hence we observe lot of full stops being generated. 
4. The repeated attention to the final sections along can be fixed by further training and by using a coverage model which would also take care of other repetitions.
5. Improvement in translation of words can be observed as compared to Normal Sequence to Sequence Model in the Attention Based Model.

## Effective Approaches to Attention-based Neural Machine Translation

1. Includes three models : Dot Based, General, Concat Based.






