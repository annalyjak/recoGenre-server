# RecoGenre – backend

The following project was implemented as part of the project "Stream Data Processing" (pl. „Przetwarzanie danych strumieniowych”) at the Wrocław University of Science and Technology.
In this repository you can find a backend: REST API and model created by [jsalbert]( https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Before you run app – you need to install all the requirements. You can do this using pip3: 

```
pip3 install -r requirements.txt
```

### Installing

To get a development env running – all you need is:

```
python3 rest.py
```

And now you can try it by typing: 

```
0.0.0.0:60022/predict
```

... in your borwser. 

You should get data in format attached below (example JSON):
```json
[  
   {  
      "genre":"classical",
      "pred":"69.281 "
   },
   {  
      "genre":"pop",
      "pred":"20.634 "
   },
   {  
      "genre":"hiphop",
      "pred":"2.808 "
   },
   {  
      "genre":"disco",
      "pred":"2.107 "
   },
   {  
      "genre":"blues",
      "pred":"1.350 "
   },
   {  
      "genre":"jazz",
      "pred":"1.341 "
   },
   {  
      "genre":"reggae",
      "pred":"0.964 "
   },
   {  
      "genre":"country",
      "pred":"0.682 "
   },
   {  
      "genre":"rock",
      "pred":"0.610 "
   },
   {  
      "genre":"metal",
      "pred":"0.222 "
   }
]
```


## Built With

* [Model jsalbert]( https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning) – CNN music genre classification model
* [Python]( https://docs.python.org/3/) – Python 3 documentation
* [Flask]( http://flask.pocoo.org/) - Used to generate REST API

## Authors

* **Anna Łyjak ** - *Initial work* - [kirjava113](https://github.com/kirjava113)

See also the list of [contributors](https://github.com/kirjava113/recoGenre-server/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* T. Lidy, C. S. (2010, April). On the suitability of state-of-the-art music information retrieval methods for analyzing, categorizing and accessing non-western and ethnic music collections. Signal Processing, strony 1032-1048.
* S. Lippens, J. M. (2004). A comparison of human and automatic musical genre classification. IEEE International Conference on Audio, Speech and Signal Processing, IV-233-IV-236.
* Y.M.G.Costa, L.S.Oliveira, A.L.Koerich, F.Gouyon i J.G.Martins. (2012, November). Music genre classification using LBP textural features. Signal Processing, 2723-2737.
* P.Kozakowski, B.Michalak. (2016, October). Music Genre Recognition. http://deepsound.io/music_genre_recognition.html
* S.Dieleman. (2014, August). Recommending music on Spotify with deep learning. http://benanne.github.io/2014/08/05/spotify-cnns.html
* M.R. French, R. H. (2007). Spectrograms: turning signals into pictures. Journal of Engineering Technology, 32-35.
* T. Ojala, M. P. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 971-987.
* M. Wu, Z. C. (2011). Combining visual and acoustic features for music genre classification. 10th International Conference on Machine Learning and Applications, 124-129.
* N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, P. T. P. Tang (2017, February). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.
