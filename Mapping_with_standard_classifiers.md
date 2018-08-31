So as a result if we want to reproduce the standard optimizers we have
to use the following parameters of universal optimizer:

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Optimizer name**                          **Params of UOptimizer to enable required optimizer \[1\]**   **Params of UOptimizer to fine-tune the optimizer (with the default value in** **brackets) \[2\]**
  ------------------------------------------- ------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------
  Vanilla SGD                                 -                                                             -

  SGD with momentum                           -   use\_exp\_avg\_norm\_type =True                           -   betas – first argument of tuple (default value is 0.9). You can pass it like (x, 0), where x is the target value.
                                                                                                            
                                                                                                            -   beta1\_dump (None)
                                                                                                            

  SGD with Nesterov momentum                  -   use\_exp\_avg\_norm\_type =True                           
                                                                                                            
                                              -   exp\_avg\_norm\_type = ‘nesterov’                         
                                                                                                            

  Adadelta                                    -   use\_exp\_avg\_sq\_norm = True                            -   betas – second argument of tuple (default value is 0.99). You can pass it like (0, x), where x is the target value
                                                                                                            
                                              -   use\_adadelta\_lr=True                                    -   eps (1e-8)
                                                                                                            
                                              -   lr=1 (recommended)                                        
                                                                                                            

  RMSProp                                     -   use\_exp\_avg\_sq\_norm = True                            -   
                                                                                                            

  Adam                                        -   use\_exp\_avg\_norm=True,                                 -   betas in form of tuple (β1, β 2). Default values are (0.9, 0.99)
                                                                                                            
                                              -   use\_exp\_avg\_sq\_norm = True,                           -   eps (1e-8)
                                                                                                            
                                              -   use\_bias\_correction= True                               
                                                                                                            

  Adamax                                      -   use\_exp\_avg\_norm=True,                                 -   
                                                                                                            
                                              -   use\_exp\_avg\_sq\_norm = True,                           
                                                                                                            
                                              -   use\_bias\_correction= True                               
                                                                                                            
                                              -   exp\_avg\_sq\_norm\_type ='infinite\_l'                   
                                                                                                            

  Nadam                                       -   use\_exp\_avg\_norm=True,                                 -   
                                                                                                            
                                              -   use\_exp\_avg\_sq\_norm = True,                           
                                                                                                            
                                              -   use\_bias\_correction= True                               
                                                                                                            
                                              -   exp\_avg\_norm\_type='nesterov'                           
                                                                                                            

  [AdamW](https://arxiv.org/abs/1711.05101)   -   use\_exp\_avg\_norm=True,                                 -   
                                                                                                            
                                              -   use\_exp\_avg\_sq\_norm = True,                           
                                                                                                            
                                              -   use\_bias\_correction= True                               
                                                                                                            
                                              -   decouple\_wd=True                                         
                                                                                                            
                                              -   set weight\_decay &gt;0                                   
                                                                                                            

  Amsgrad                                     -   use\_exp\_avg\_norm=True,                                 -   
                                                                                                            
                                              -   use\_exp\_avg\_sq\_norm = True,                           
                                                                                                            
                                              -   use\_bias\_correction= True                               
                                                                                                            
                                              -   exp\_avg\_sq\_norm\_type = 'max\_past\_sq'                
                                                                                                            
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

\[1\] We note here only the parameters which you have to change from
default values.

\[2\] In order to avoid repeating , in all cases you can change default
values of learning rate (parameter “lr”, default value is 1e-3) and
weight decay regularization (“weight\_decay”, default value is zero)
