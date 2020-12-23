# translator

Используем пакеты `transformers` , `pytorch` и другие.

Simple run predict for translator OPUS:

``` python
translator import OPUSModel

tr = OPUSModel("Helsinki-NLP/opus-mt-en-ru")

output = tr.predict("My name is Sergey and I live in Stockgolm")
#  '<pad> Меня зовут Сергей и я живу в России.'
print(output)

input_text = "My name is Wolfgang and I live in Berlin"                                                                                                     

target_text = "Меня зовут Вольфганг и я живу в Берлине."                                                                                                    

loss = tr.loss(input_text=input_text, target_text=target_text)                                                                                              

print(f"Loss: {loss}")
```
