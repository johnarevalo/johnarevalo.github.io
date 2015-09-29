---
layout: post
title: "Just another char-rnn generator model using Blocks and spanish lyrics"
---

The [Andrej's blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
gained a lot of popularity because of their interesting results generating text
in several scenarios. Taking a single txt file as input, he was able to
automatically generate char-by-char from Shakespeare's dialogs to C++ code.
The strategy was also used to [generate
music](https://highnoongmt.wordpress.com/2015/05/22/lisls-stis-recurrent-neural-networks-for-folk-music-generation/)
and [bible's phrases](http://cpury.github.io/learning-holiness/). Those results
come from learning a language model using recurrent neural networks (RNN).

This post is just another exploration of such kind of models. This time is the
turn for spanish reggaeton lyrics. I think it could be fun to train a language
model on this genre because of its strongly inclination to messy (and sometimes
also grose) speech in many of its lyrics. You can directly go to the [generated
text](#composing-songs), or read below a bit of technical details.

## About the implementation

The learning-by-doing approach works for me. So, I wanted to reimplement the
original [Torch version](https://github.com/karpathy/char-rnn) using the
[Blocks](http://blocks.readthedocs.org/) framework.

Blocks is a very useful library which makes easier to build complex neural
network architectures with [Theano](http://deeplearning.net/software/theano).
Blocks already brings a ready-to-use SequenceGenerator that includes additional
components such as AttentionModel and Emitters. Since my goal was to better
explore and understand the details of generative RNN models (preprocessing,
sampling, etc), I decided to keep it as simple as possible, by mimicking the
original code and implementing the sampling process by my own. The implemented
code is available on [GitHub](https://github.com/johnarevalo/blocks-char-rnn).

## Training procedure

I took 241 reggaeton artist from [freebase](http://www.freebase.com/m/0233qs),
and joined them with around 100.000 lyrics songs crawled from the web, resulting
in 6800 reggaeton songs with 11 millions of characters. I trained a RNN with 3
layers, with 512 GRU units each one, yielding to 5 millions parameters in total.
I didn't use dropout. The training took around 3 hours for 10 epochs.

## Composing songs

I generated a [50K characters sequence](https://goo.gl/BMIooc). This is an
interesting chunk of text:

{% highlight text %}
un pin record,
criminal...mahoe...
una nueva, me gusta cuando fue tu tiempo
(muevelo salio de este corazón, que ninguna puede)
regresa
sacudela, lo cuando piensas conmigo
no quiero escapar con construyudo y adelantar
una ves mas..
usted no frontea,
tengo gafas nunka
veras que aunque no pueden olvidar
que lloren y sepas
tus ojos seran tu party (pitb)
me llevas o te amo[x2)

unos he hecho
tu voz con balas maldy
si eh llevado y contenerse el mahon
ya tengo a la disco
siempre quiere eh echo

en puerto rico no es una chica chica malvada
zona dice la amarrula y via que ya no estuvo maja
y ni siquiera demás
es que tu le decias de mi haciendo, sus derras y las luces,
tienes que dormir
ya tu me gustas,despues por que me arrepiento
pero mami no te mandas a case mi significa
no te pongas histir a mi, en la cama no se toca
pa ver quien dijilda...
alex genki
like

quiero que esté hip-hop?
bayamogregame opetarla lo que arranquila
siente el remix, _______ pa' mi
que ven, perra, pa'ca, pa’ cas sanso
pa' comprenderse, y no voy a parar
disparo borracho tapues a buscarte

esa gata maquinon por lo que yo quiero es bella
mono mañana abachao'pela'o hahaaa
que las carreras estan vivo venimos en la navidad

[daddy yankee]

vamo’ a ver quien soy yo
father, jaja
what! what?
'taele calor, si el niños no me coja
ya tu mas pilla que pro-titer
eval - daddy-l-rit
the know traquee-l como
you know the champion
..eh... oculto...uy,teligo..
bacambiando rodando....
sentirte.

muke ha sido deciratira
de jugarte de su hombre

chiquillo, como el culo se activara
y no es nuestro gran
olvidar y es un sex
quemando mis rodilles emegan los errores
libre libre libre libre libre libre desde los kono block mr abasta!
voy a sandunga con mayor fata
los chuletas bucha de la cama
como tu me hables
nada no se haya olvidado
y no la gusta
en mi mente derrecuelca la pase con miguelito
pa que yo puedo evitar
escucha el mundo entero
que yo sigo en silencio
ya lo hago pa aira si cuando eutola:
¿se mueve? que te llevo a todas quieren hacerla con ganas
que no se apinta dejas lo que yo no te quiera
nadie te ama pasa a ti,
llegar los pensamientos guerreros andan bolisos
mueve rastrilla dale caliente (ah ah ah ah ah ah ah ah ah)
ahora escuche es un día del amor,
somos tu y yo (eso es asi
como loco y ojo por lo de hooque este ooo

ra-ra lo recito con tocarte
(dale duro microu)
te llamo (hey!)
(randy)
20 no vengan rodq//i please, pero yo y en el perreo guayando guaya
los tecatos reggaeton y te tocaba pa' la vida pensando
que es una barria porque
desde que te vi, se monta
me da la razon,
que la monte yo ando solo desatruma, e solo decir
que me tiene envidia,
y no confió en mi soldado y tú me siento prevideo
ya no sale en ti creada liriqueado
por el sufrimiento pierdo el control uoo
desenfrenado, y de sabana buscando
que conmigo se to
en tu cachorro, nena, ma', novia me enseño
cuando bailar bien mi nena (á su cuerpo baila)
pegate mami que yo bendito vamo a montarla
aqui kendo en las voz sin nautarina)
y en la fuerza en un barrio
de duro, lo q sueño solo
ono, falta la velo no me apures no fui yo fuera intento mi chamaquito a los hommieron
luna llego y te entregaré como el fue pa mi pícur

{% endhighlight %}

Once more, this kind of models shows promising results. It exposes the
general structure of the corpus, i.e. short lines grouped in paragraphs (like
strophes). Also, the model learned to open and close correctly brackets and
parenthesis. It also learn most of the syntactic rules in spanish language (at
least those that the training corpus exposes), whose usually are more complex
than english ones (verb conjugation, accentuation, additional punctuation, etc).

But more interestingly, from time to time the model decides to include the
`[coro]` tag indicating the chorus, mention some artist between brackets, or
even include some english phrases, just like usual reggaeton songs do.

On the downside, it generates several misspellings and, at the end, the generated
text does not have a concrete story. However, we shouldn't be worried about it
because anyway: it is reggaeton lyrics!

{% include disqus.html %}
