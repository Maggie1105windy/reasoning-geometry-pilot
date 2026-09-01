# Stage A2 Gate 4 — Full Activation Integrity

Status: **PASS**

## Storage

- 78 atomic/resumable shards, each with separate layer files
- Official `adjust_precision(..., output_precision=8, per_channel=True)`
- Storage dtype/qscheme: `torch.quint8` / `torch.per_channel_affine_float_qparams`, axis 1
- Quantization parameters are fit independently per shard and layer
- Probe inputs are dequantized to float32

## Shape and finite-value checks

```text
 layer                      hook         shape  row_count  hidden_dim storage_dtype dequantized_dtype  nan_count  inf_count
     0  blocks.0.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     1  blocks.1.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     2  blocks.2.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     3  blocks.3.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     4  blocks.4.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     5  blocks.5.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     6  blocks.6.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     7  blocks.7.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     8  blocks.8.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
     9  blocks.9.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    10 blocks.10.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    11 blocks.11.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    12 blocks.12.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    13 blocks.13.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    14 blocks.14.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    15 blocks.15.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    16 blocks.16.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    17 blocks.17.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    18 blocks.18.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    19 blocks.19.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    20 blocks.20.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    21 blocks.21.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    22 blocks.22.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    23 blocks.23.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    24 blocks.24.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    25 blocks.25.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    26 blocks.26.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    27 blocks.27.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    28 blocks.28.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    29 blocks.29.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    30 blocks.30.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
    31 blocks.31.hook_resid_post [39585, 2560]      39585        2560  torch.quint8     torch.float32          0          0
```

## Ten-row live-forward comparison

```text
 sample_order  row_id                                                entity                                                                                    prompt  entity_last_token_index entity_last_token  layer      mae  max_absolute_error  cosine_similarity
            0    3401                                   San Gregorio Armeno                                   What are the lat/lon coordinates of San Gregorio Armeno                       13               eno      0 0.002386            0.018198           0.999992
            0    3401                                   San Gregorio Armeno                                   What are the lat/lon coordinates of San Gregorio Armeno                       13               eno      7 0.004442            0.012914           0.999987
            0    3401                                   San Gregorio Armeno                                   What are the lat/lon coordinates of San Gregorio Armeno                       13               eno     15 0.005503            0.134560           0.999992
            0    3401                                   San Gregorio Armeno                                   What are the lat/lon coordinates of San Gregorio Armeno                       13               eno     23 0.009734            0.163597           0.999995
            0    3401                                   San Gregorio Armeno                                   What are the lat/lon coordinates of San Gregorio Armeno                       13               eno     31 0.027640            0.358452           0.999950
            1   30630                                     San Mauro Pascoli                                     What are the lat/lon coordinates of San Mauro Pascoli                       13              coli      0 0.002523            0.018994           0.999991
            1   30630                                     San Mauro Pascoli                                     What are the lat/lon coordinates of San Mauro Pascoli                       13              coli      7 0.004786            0.027723           0.999988
            1   30630                                     San Mauro Pascoli                                     What are the lat/lon coordinates of San Mauro Pascoli                       13              coli     15 0.005571            0.020037           0.999992
            1   30630                                     San Mauro Pascoli                                     What are the lat/lon coordinates of San Mauro Pascoli                       13              coli     23 0.008999            0.125488           0.999996
            1   30630                                     San Mauro Pascoli                                     What are the lat/lon coordinates of San Mauro Pascoli                       13              coli     31 0.031404            0.312252           0.999923
            2    3532        Lexington, Kentucky's Christ Church Cathedral         What are the lat/lon coordinates of Lexington, Kentucky's Christ Church Cathedral                        17                 Ġ      0 0.002380            0.011098           0.999992
            2    3532        Lexington, Kentucky's Christ Church Cathedral         What are the lat/lon coordinates of Lexington, Kentucky's Christ Church Cathedral                        17                 Ġ      7 0.004417            0.014017           0.999987
            2    3532        Lexington, Kentucky's Christ Church Cathedral         What are the lat/lon coordinates of Lexington, Kentucky's Christ Church Cathedral                        17                 Ġ     15 0.005363            0.018607           0.999993
            2    3532        Lexington, Kentucky's Christ Church Cathedral         What are the lat/lon coordinates of Lexington, Kentucky's Christ Church Cathedral                        17                 Ġ     23 0.008544            0.120834           0.999995
            2    3532        Lexington, Kentucky's Christ Church Cathedral         What are the lat/lon coordinates of Lexington, Kentucky's Christ Church Cathedral                        17                 Ġ     31 0.021726            0.317648           0.999925
            3   25906                                       India's Belonia                                       What are the lat/lon coordinates of India's Belonia                       12              onia      0 0.002584            0.016973           0.999989
            3   25906                                       India's Belonia                                       What are the lat/lon coordinates of India's Belonia                       12              onia      7 0.004821            0.047794           0.999988
            3   25906                                       India's Belonia                                       What are the lat/lon coordinates of India's Belonia                       12              onia     15 0.005788            0.076469           0.999992
            3   25906                                       India's Belonia                                       What are the lat/lon coordinates of India's Belonia                       12              onia     23 0.008984            0.099998           0.999996
            3   25906                                       India's Belonia                                       What are the lat/lon coordinates of India's Belonia                       12              onia     31 0.019890            0.150917           0.999972
            4   17370                               Virginia's Lowes Island                               What are the lat/lon coordinates of Virginia's Lowes Island                       13           ĠIsland      0 0.002501            0.013909           0.999990
            4   17370                               Virginia's Lowes Island                               What are the lat/lon coordinates of Virginia's Lowes Island                       13           ĠIsland      7 0.004810            0.042278           0.999988
            4   17370                               Virginia's Lowes Island                               What are the lat/lon coordinates of Virginia's Lowes Island                       13           ĠIsland     15 0.005904            0.030851           0.999993
            4   17370                               Virginia's Lowes Island                               What are the lat/lon coordinates of Virginia's Lowes Island                       13           ĠIsland     23 0.008602            0.080879           0.999997
            4   17370                               Virginia's Lowes Island                               What are the lat/lon coordinates of Virginia's Lowes Island                       13           ĠIsland     31 0.025381            0.210332           0.999948
            5   17138                                                Shklow                                                What are the lat/lon coordinates of Shklow                       11               low      0 0.002556            0.024235           0.999989
            5   17138                                                Shklow                                                What are the lat/lon coordinates of Shklow                       11               low      7 0.004886            0.015033           0.999986
            5   17138                                                Shklow                                                What are the lat/lon coordinates of Shklow                       11               low     15 0.005922            0.067402           0.999991
            5   17138                                                Shklow                                                What are the lat/lon coordinates of Shklow                       11               low     23 0.008604            0.240440           0.999995
            5   17138                                                Shklow                                                What are the lat/lon coordinates of Shklow                       11               low     31 0.026901            0.356861           0.999924
            6   27603                                   Hampshire's Bramley                                   What are the lat/lon coordinates of Hampshire's Bramley                       13               ley      0 0.002536            0.020741           0.999988
            6   27603                                   Hampshire's Bramley                                   What are the lat/lon coordinates of Hampshire's Bramley                       13               ley      7 0.004824            0.015606           0.999987
            6   27603                                   Hampshire's Bramley                                   What are the lat/lon coordinates of Hampshire's Bramley                       13               ley     15 0.005716            0.017540           0.999991
            6   27603                                   Hampshire's Bramley                                   What are the lat/lon coordinates of Hampshire's Bramley                       13               ley     23 0.008379            0.071327           0.999996
            6   27603                                   Hampshire's Bramley                                   What are the lat/lon coordinates of Hampshire's Bramley                       13               ley     31 0.023926            0.480515           0.999913
            7    3728 Los Angeles's St. Thomas the Apostle Catholic Church  What are the lat/lon coordinates of Los Angeles's St. Thomas the Apostle Catholic Church                        20                 Ġ      0 0.002353            0.023208           0.999991
            7    3728 Los Angeles's St. Thomas the Apostle Catholic Church  What are the lat/lon coordinates of Los Angeles's St. Thomas the Apostle Catholic Church                        20                 Ġ      7 0.004195            0.027650           0.999988
            7    3728 Los Angeles's St. Thomas the Apostle Catholic Church  What are the lat/lon coordinates of Los Angeles's St. Thomas the Apostle Catholic Church                        20                 Ġ     15 0.005213            0.045341           0.999994
            7    3728 Los Angeles's St. Thomas the Apostle Catholic Church  What are the lat/lon coordinates of Los Angeles's St. Thomas the Apostle Catholic Church                        20                 Ġ     23 0.009996            0.239708           0.999995
            7    3728 Los Angeles's St. Thomas the Apostle Catholic Church  What are the lat/lon coordinates of Los Angeles's St. Thomas the Apostle Catholic Church                        20                 Ġ     31 0.036959            0.357353           0.999871
            8    7974                                    Chancheng District                                    What are the lat/lon coordinates of Chancheng District                       12         ĠDistrict      0 0.002237            0.014246           0.999993
            8    7974                                    Chancheng District                                    What are the lat/lon coordinates of Chancheng District                       12         ĠDistrict      7 0.004343            0.056374           0.999992
            8    7974                                    Chancheng District                                    What are the lat/lon coordinates of Chancheng District                       12         ĠDistrict     15 0.005298            0.024139           0.999995
            8    7974                                    Chancheng District                                    What are the lat/lon coordinates of Chancheng District                       12         ĠDistrict     23 0.008625            0.180973           0.999996
            8    7974                                    Chancheng District                                    What are the lat/lon coordinates of Chancheng District                       12         ĠDistrict     31 0.023235            0.458336           0.999959
            9   33984                                              El Jebha                                              What are the lat/lon coordinates of El Jebha                       12                ha      0 0.002539            0.019669           0.999989
            9   33984                                              El Jebha                                              What are the lat/lon coordinates of El Jebha                       12                ha      7 0.004896            0.027573           0.999985
            9   33984                                              El Jebha                                              What are the lat/lon coordinates of El Jebha                       12                ha     15 0.005921            0.056984           0.999991
            9   33984                                              El Jebha                                              What are the lat/lon coordinates of El Jebha                       12                ha     23 0.008526            0.065689           0.999996
            9   33984                                              El Jebha                                              What are the lat/lon coordinates of El Jebha                       12                ha     31 0.038804            0.311768           0.999900
```

- Minimum cosine similarity: 0.999870837
- Mean MAE: 0.00984151762
- Maximum absolute error: 0.480514526

## Checklist

- [x] manifest_rows_39585
- [x] activation_rows_39585_each_layer
- [x] label_rows_39585
- [x] layer_count_32
- [x] hidden_dim_2560
- [x] train_count_31668
- [x] test_count_7917
- [x] no_nan
- [x] no_inf
- [x] row_id_original_order
- [x] all_pre_forward_integrity_checks
- [x] live_forward_min_cosine_gt_0.999
