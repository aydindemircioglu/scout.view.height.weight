
# Determining body height and weight from thoracic and abdominal CT localizers in pediatric and young adult patients using deep learning

This is the code repository for the paper, published in Scientific Reports,
https://doi.org/10.1038/s41598-023-46080-5

The code is not polished and will need some work to get it running.
No data is included, and one needs to infer from the code where to put
the data. The preprocessing routines used are in ./prepareData.py.

No requirements.txt was added, it should work fine with the
current libraries (April 2023).


## Training

Start training of the two different methods in each of the directories
Standard and Standard_no_pretrain. The difference between both is just
that Standard uses the young adult training set additionally.
To train, first, one starts train_optuna.py. It will use Optuna to
tune the hyperparameters. The best model is then retrained (retrain.py)
and stopped early. However, this model now uses all data for training, and no longer
 validates the model (the validation set is the training set, so its fake). Finally, the retrained model will be tested on
the test data once by calling reevaluate.py




## LICENCE

Copyright 2023, Aydin Demircioglu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
