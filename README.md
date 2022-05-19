## ZHAW School of Management and Law

### Statistik WIN, 2022-FS
### Dozenten: A. Fazlija, M. Schnauss
### 19 Mai 2022

## Vortragsthema 03: SMI Analyse

### Namen der Gruppenmitglieder:

### Gregory Hill        hillgre1@students.zhaw.ch

### Michael Neukom      neukomic@students.zhaw.ch

### Tobias Fatzer       fatzetob@students.zhaw.ch

### Codeverwaltung:	[Github](https://github.com/TobiasFatzer/STAT_Abgabe)

### Dokumentation:	[Webseite](https://tobiasfatzer.github.io/STAT_Abgabe/)


## 1	Aufgabenstellung
### 1.1	Teilaufgaben

(a)	Bestimmen Sie den Median der empirischen Renditeverteilung.

(b)	Bestimmen Sie das erste Quartil der Renditeverteilung.

(c)	Das arithmetische Mittel der diskreten Renditen wird berechnet, falls die erwartete Rendite in zukünftigen Monaten (ex ante) zu schätzen ist. Bestimmen Sie dieses mit den verfügbaren Daten.

(d)	Das geometrische Mittel diskreter Renditen wird berechnet, falls die tatsächliche Performance (ex post) zu ermitteln ist. Bestimmen Sie dieses mittels Anfangs- und Endwert des SMI.

(e)	Schätzen Sie die Standardabweichung (Volatilität) der SMI-Monatsrenditen und simulieren Sie den weiteren Verlauf des SMI.

### 1.2	Rahmenbedingungen

Folgende Tabelle stellt die Monatsperformance des SMI-Index von Mai 2020 bis April 2022 dar. Wir haben uns dazu entschieden, diese Tabelle über den möglichen Dateiexport herunterzuladen.

<img src="images\Tabelle_der_Daten.png"/>

Tabelle 1: Monatsperformance des SMI-Index



#### 1.3	Annahmen
Wir nahmen während dieser Arbeit an, dass es sich bei unserem Datensatz um die Population und keine Stichprobe handelt. Jedoch wurden wir letztens in einer Grossklasse verunsichert, da erwähnt wurde, dass die «Schätzungen der Standardabweichungen» nur bei Stichproben stattfinden können. Folgend wurden alle, in Python gelösten Aufgaben, mit der Annahme gelöst, dass es sich um eine Population handelt.

## 2	Ausarbeitung in Python
### 2.1	Libraries
Zur Bearbeitung der Aufgaben a – e wurden folgende Libraries verwendet:


    import os
    import statistics
    from datetime import datetime
    import matplotlib.pyplot as plt
    import pandas as pd
    from dateutil.relativedelta import relativedelta

Die Library «os» ermöglicht uns, den Code unabhängig vom Betriebssystem auszuführen. Dies war in unserem Projekt relevant, da sowohl mit macOS sowie Windows gearbeitet wurde. Insbesondere unterscheiden sich die Betriebssysteme beim Pfad zur benötigten Datei. Weitere Erläuterungen sind im Kapitel Datenbearbeitung enthalten.

Mit der Library «statistics» können wir verschiedene Werte aus der Statistik berechnen. Dazu gehören unter anderem der Median, das arithmetische- , sowie das geometrische Mittel und der Modus.

«Pandas» wird für die Datenverarbeitung benötigt. Ein häufiger Gebrauch von «Pandas» ist das Erstellen von Tabellen.

Mit «datetime» ist es uns möglich, mit Datum- und Zeitangaben zu rechnen.

«relativedelta» ist eine Erweiterung zu «datetime» und kann mit relativen Werten arbeiten. Zum Beispiel vorheriger Tag oder nächster Monat.

Die Libraries die für die Bearbeitung der Aufgabe d verwendet wurden, werden im Kapitel 2.8.3 erläutert.


### 2.2	Datenbearbeitung
In diesem Kapitel wird die Datenverarbeitung genauer umschrieben. Hierbei handelt es sich um den folgenden Code:

    #os.path.dirname(os.path.realpath('SMI_Historical_Data.csv') gets local path to this file and then adds the File to path so File is selected
    dataframe = pd.read_csv(os.path.dirname(os.path.realpath('SMI_Historical_Data.csv')) + '/SMI_Historical_Data.csv')

    #Gets Column Price and replaces the "," with nothing so that the value can be read as float, then gets cast to float and written into Close column; Same goes for column Change %
    dataframe['Close'] = dataframe['Price'].str.replace(',', '').astype(float)
    dataframe['Change %'] = dataframe['Change %'].str.replace('%', '').astype(float)

    #Turns around the dataset, since the dataset int he csv File is somehow in the wrong way around (The Closest Date is First)
    dataframe = dataframe.iloc[::-1]
    
    #Iterates through Date Column and replaces existing value with datetime value
    datelist = []
    for i in range(0, len(dataframe.index)):
       datelist.append(datetime.strptime(str(dataframe['Date'][i]), '%b %y'))
    
    #Puts the Column Date as Index for the Dataframe
    datelist = datelist[::-1]
    dataframe.index = datelist
    
    #Drops Column Date since It's now the index and drop column Price to prevent confusion
    dataframe.drop('Date', axis=1, inplace=True)
    dataframe.drop('Price', axis=1, inplace=True)
    
    
    dataframe

#### 2.2.1	CSV einlesen und Pfad festlegen
    dataframe = pd.read_csv(os.path.dirname(os.path.realpath('SMI_Historical_Data.csv')) + '/SMI_Historical_Data.csv')

In einem ersten Schritt wird die CSV Datei eingelesen. Um die Flexibilität und Funktionalität zu gewährleisten wird das File mit «os.path» aufgerufen. So können unabhängig vom Dateisystem und dem Standort der CSV Datei die Daten abgerufen werden. Das bedeutet, wenn das Verzeichnis in Windows im Downloadordner abgelegt ist, wird zuerst der Pfad “C:\Users\User\Downloads” abgerufen und anschliessend der Dateiname angefügt. So wird der Pfad auf “C:\Users\User\Downloads\SMI_Historical_Data.csv” festgelegt.

#### 2.2.2	Daten zu float Werten umwandeln
    dataframe['Close'] = dataframe['Price'].str.replace(',', '').astype(float)
    dataframe['Change %'] = dataframe['Change %'].str.replace('%', '').astype(float)

In diesem Schritt werden die Daten angepasst, damit sie für unsere Libraries lesbar werden. In der ersten Zeile wird das “,” eliminiert und anschliessend der Wert als float definiert. In der zweiten Zeile wird das “%” Symbol wiederum auch eliminiert und der Wert erneut als float festgelegt. 

Wir definieren diese Daten als float, um mit Ihnen, sobald sie in die dataframes abgefüllt wurden, Kalkulationen anzustellen. So können wir nur die Daten bearbeiten, welche nötig sind, um die gewünschten Ergebnisse zu erhalten.


#### 2.2.3	Werte nach “Älteste Zuerst” sortieren
    dataframe = dataframe.iloc[::-1]

Hier werden die Daten anhand des Erfassungsdatums von Älteste nach Neuesten sortiert, da im CSV die Werte von Neuste nach Ältesten sortiert sind und die Verarbeitung so erschwert wird.

#### 2.2.4	Datum der Werte anpassen
    datelist = []
    for i in range(0, len(dataframe.index)):
       datelist.append(datetime.strptime(str(dataframe['Date'][i]), '%b %y'))

In dieser Code Zeile wird zuerst ein neues Array erstellt, um die korrekten Zeitdaten zu erfassen. Anschliessend legen wir die Länge des Arrays auf die Menge der Daten im dataframe fest. Danach formen wir die vorhandenen Zeit Daten in das konventionelle, von Python lesbare Zeit Datums Format um.

#### 2.2.5	Datum als Indexwert definieren
    datelist = datelist[::-1]
    dataframe.index = datelist

In diesem Teil sortieren wir wieder wie in 2.2.3 die Daten nach Ältesten zuerst. Anschliessend legen wir das neue Array aus 2.2.4 als Index für unser dataframe fest. Somit haben wir alle nötigen Daten erfasst und lesbar in einer Tabelle (dataframe) festgehalten.
#### 2.2.6	Bereinigen
    dataframe.drop('Date', axis=1, inplace=True)
    dataframe.drop('Price', axis=1, inplace=True)

Um die Lesbarkeit noch zu erhöhen, entfernen wir in diesem Teil die nicht mehr benötigten Spalten. In diesem Fall handelt es sich um die Datumsangaben und um den Preis des SMI vor der Umwandlung.
Nun werden die Daten wie in der Abbildung 1 angezeigt.

<img src="images\Dataset_nachbearbeitung.png"/>

Abbildung 1: Datenset nach der Datenverarbeitung

### 2.3	Daten mit Pandas Plotten
Um die in Abbildung 1 gezeigten Daten übersichtlicher darzustellen, verwenden wir eine Kombination aus Pandas und matplotlib. Das in Abbildung 1 gezeigte Dataframe kann mit folgendem Code zu einem Boxplot umgewandelt werden:

    dataframe['Renditeverteilung'].plot(kind='box', grid=True, title='BoxPlot Renditeverteilung SMI', color='black')

Es können auch sehr komplexe Graphen erstellt werden, bei denen Attribute wie Titel, Achsen und Legende angepasst werden können. Ein Beispiel dafür zeigt uns folgender Code:

    plt.plot(dataframe['Renditeverteilung'], color='#42bbf7', label='Renditeverteilung')
    plt.title('Renditeverteilung 2y')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel('Percental Change')

    plt.axhline(y=dataframe['Renditeverteilung'].median(), color='yellow', label='Median')
    plt.axhline(y=dataframe['Renditeverteilung'].describe()['25%'], color='green', label='First Quartile')
    plt.axhline(y=round(dataframe['diskrete Rendite'].mean(), 2), color='blue', label='Mean')
    plt.axhline(y=round(geomid, 2), color='pink', label='Geometric mean')
    
    plt.legend()
    plt.show()

Dieser Code zeigt uns einen Graphen, der die Teilaufgaben a, b und c bereits beantwortet und in Abbildung 2 gezeigt ist.

<img src="images\Complicated_plots.png"/>

Abbildung 2 - Komplexe Graphen mit Pandas


### 2.4	Aufgabe a - Bestimmen Sie den Median der empirischen Renditeverteilung. 
Zuerst wird die Renditeverteilung mittels der Funktion dataframe erstellt. Dabei wird das Erstellte Datenset benutzt.

    dataframe['Renditeverteilung'] = round(dataframe['Close'] * 100 / dataframe['Close'].shift() - 100, 2)

Als nächstes werden alle Null- und NaN-Werte aus der neuen Tabelle mit den korrekten Daten ersetzt. Dieser Schritt war nötig, um einerseits die Daten zu bereinigen und anderseits eine Überprüfung der erhaltenen Werte durchzuführen.

    if dataframe['Renditeverteilung'].isnull().values.any() or dataframe['diskrete Rendite'].isnull().values.any():
        dataframe["Renditeverteilung"].fillna(dataframe['Change %'], inplace=True)

Nun kann mithilfe der «statistics» Library der Median ausgelesen werden.

    dataframe['Renditeverteilung'].median()

Folgende Abbildung zeigt den Boxplot der Renditeverteilung, der Median ist in Rot gekennzeichnet.

<img src="images\Boxplot_median.png"/>

Abbildung 3 - BoxPlot mit Median in Rot

### 2.5	Aufgabe b - Bestimmen Sie das erste Quartil der Renditeverteilung

Analog zur Berechnung des Medians, wird auch bei dieser Aufgabe eine Funktion aus der «statistics» Library verwendet.

    dataframe['Renditeverteilung'].describe()['25%'])

Folgende Abbildung zeigt den Boxplot der Renditeverteilung, das Erste Quartil ist in Rot gekennzeichnet.

<img src="images\Boxplot_Quartil.png"/>

Abbildung 4 - BoxPlot mit Erstem Quartil in Rot

#### 2.6	Aufgabe c - Das arithmetische Mittel der diskreten Renditen wird berechnet, falls die erwartete Rendite in zukünftigen Monaten (ex ante) zu schätzen ist. Bestimmen Sie dieses mit den verfügbaren Daten.
Zuerst wird die diskrete Rendite berechnet.

    dataframe['diskrete Rendite'] = (dataframe['Close'] / dataframe['Close'].shift() - 1) * 100

Anschliessend werden die Null- und NaN-Werte auf die gleiche Art und Weise wie in Kapitel 2.2.5 durch die korrekten Daten ersetzt. Zum Schluss wird das Arithmetische Mittel ausgerechnet. 

    dataframe['diskrete Rendite'].mean()


### 2.7	Aufgabe d - Das geometrische Mittel diskreter Renditen wird berechnet, falls die tatsächliche Performance (ex post) zu ermitteln ist. Bestimmen Sie dieses mittels Anfangs- und Endwert des SMI.

    geomid = pow(dataframe['diskrete Rendite'][0] * dataframe['diskrete Rendite'][len(dataframe['diskrete Rendite']) - 1], 1 / 2)

Um das geometrische Mittel in Python berechnen zu können, müssen wir zuerst eine Formel dafür definieren. Dies aus dem Grund, da die Hauseigene Funktion von der Library «statistics» nicht unsere Aufgabenstellung abdeckt. Deshalb haben wir die Funktion «geomid» erstellt, welche wir wie die anderen Funktionen anwenden können. Die Formel haben wir gemäss Aufgabenstellung mit dem Anfangs- und Endwert aufgestellt. Entsprechend weicht sie stark vom arithmetischen Mittel aus Kapitel 2.6 ab. Nach unseren Berechnungen beträgt das Geometrische Mittel somit 2.85%.

###  2.8	Aufgabe e - Schätzen Sie die Standardabweichung (Volatilität) der SMI-Monatsrenditen und simulieren Sie den weiteren Verlauf des SMI.
#### 2.8.1	Berechnung der Standartabweichung

Mittels der Library «statistics» und deren Funktion «pstdev» war es uns möglich die Standardabweichung einer Liste zu berechnen. Diese Liste wird aus der Kolone Renditeverteilung in der Tabelle dataframe erstellt.

    import statistics
    statistics.pstdev(dataframe['Renditeverteilung'])

Die Library «statistics» stellt zwei verschiedene Funktionen für das Berechnen der Renditeverteilung zur Verfügung. Einerseits die von uns verwendete «pstdev» Funktion, welche die Varianz der Population berechnet. Anderseits «stdev» welches die Varianz einer Stichprobenmenge berechnet. Wie bereits im Kapitel 2 erwähnt, gehen wir davon aus, dass es sich bei unserem Datensatz um eine Population handelt.

#### 2.8.2	Simple Simulationen des SMI

Um den weiteren Verlauf des SMI zu berechnen haben wir uns überlegt, dass der nächste Eintrag mit grosser Wahrscheinlichkeit zwischen der höchsten und niedrigsten Standardabweichung liegen wird. Dies würde für unsere Kalkulation bedeuten, dass der nächste Monat, zwischen 12'000 CHF und 12'900 CHF liegen wird (Abbildung 5). Tatsächlich war der Close Wert des SMI im April bei 12'128.76 CHF. 

<img src="images\Mögliche_entwicklung_des_SMIS_April.png"/>

Abbildung 5 - Mögliche Entwicklung des SMIs im Monat April

Basierend auf diesem Wissen haben wir einen Graphen erstellt, der uns den optimalsten und suboptimalsten Verlauf aufgrund der Standardabweichung des SMI, sowie den Durchschnitt (auf der Abbildung «Absolut_Median» genannt; ist ein Fehler und sollte «Absolut_Mean» heissen) darstellt (Abbildung 6)

<img src="images\optimale_smi_entwicklung.png"/>

Abbildung 6 - Optimale und Suboptimale Entwicklung des SMIs

Der folgende Code erstellt die Tabelle, die für das Darstellen der Abbildung 6 benötigt wird. Alle Werte werden erst einmal mit dem Close Wert des erhaltenen Datensatzes gefüllt, um den Verlauf besser darzustellen

    absolut_df = pd.DataFrame(dataframe['Close'])
    absolut_df['Absolut_Min_Prediction_Value'] = dataframe['Close']
    absolut_df['Absolut_mean'] = dataframe['Close']
    absolut_df['Absolut_Max_Prediction_Value'] = dataframe['Close']

Für die folgenden Einträge wurde zuerst eine prozentuale Tabelle erstellt, welche die Abweichungen Prozentual ausrechnet. Diese Tabelle wurde dann verwendet, um die absoluten Zahlen zu berechnen und in der Abbildung 6 darzustellen. Der folgende Code erstellt 10 neue Einträge und fügt diese dem dataframe „absolut_df“ hinzu. 

    for i in range(0, 10):
        tmp2 = pd.DataFrame({
             'Close':[absolut_df['Close'][len(absolut_df.index) - 1] * ((100 + df['mean'][len(df['mean']) - 1]) / 100)],
             'Absolut_Min_Prediction_Value':[absolut_df['Close'][len(absolut_df.index) - 1] * ((100 + df['Min_Prediction_Value'][len(df['Min_Prediction_Value']) - 1]) / 100)],
             'Absolut_mean':[absolut_df['Close'][len(absolut_df.index) - 1] * ((100 + df['mean'][len(df['mean']) - 1]) / 100)],
             'Absolut_Max_Prediction_Value': [absolut_df['Close'][len(absolut_df.index) - 1] * ((100 + 
        }, index=[absolut_df.index[-1] + relativedelta(months=1)])
    
    df.loc[len(df.index)] = [df['Min_Prediction_Value'][len(df['Min_Prediction_Value'])-1] - allValStandartabweichung, allValmean, df['Max_Prediction_Value'][len(df['Max_Prediction_Value']) - 1] + allValStandartabweichung]
    
    absolut_df = pd.concat([absolut_df, tmp2])

Nach Erstellung dieses Graphen konnten wir unseren ersten Vorhersagealgorithmus laufen lassen. Dieser befolgte die Regel, dass nach der Berechnung jedes neuen Close Wertes der nächste Wert nicht höher oder tiefer als die mit diesem Wert berechnete Standardabweichung sein kann. Der Code für diese Rechnung sieht wie folgt aus.

    'Random Prediction': [
       round(absolut_df['Close'][len(absolut_df.index) - 1] * ((100 + (random.uniform(df['Min_Prediction_Value'][len(df['Min_Prediction_Value']) - 1], df['Max_Prediction_Value'][len(df['Max_Prediction_Value']) - 1]))) / 100), 2)
    ]

Wie unser Graph mit diesem zusätzlichen Code aussieht, wird in der Abbildung 7 widerspiegelt.

<img src="images\erste_smi_Prediction.png"/>

Abbildung 7 - Erste Prediction des SMI verlaufs

Dies ist eine erste mögliche Simulation des SMI Verlaufs. Diese Simulation bestätigt auch unsere Hypothese, dass der optimalste und souboptimalste Wert nicht überschritten werden darf. Nach sechs dieser Vorhersagen ist aufgefallen, dass die Simulationen wenig Ähnlichkeit mit einem realistischen Verlauf des SMI aufweisen. Wie in Abbildung 8 erkennbar ist, fungieren die Extreme auch als sogenannte Resistance, wobei die Predictions an diesen abprallen. Zudem ist der vorhergesagte Verlauf zu Volatil für einen Index.

<img src="images\sechs_mögliche_SMI_vorhersagen.png"/>

Abbildung 8 - Sechs mögliche SMI Predictions

#### 2.8.3	SMI Simulation mit einem LSTM Model
Um diese Simulationen zu verfeinern und realitätsnaher darzustellen haben wir uns entschieden, im Rahmen der Arbeit uns oberflächlich mit Neuralen Netzen, Machine Learning und AI auseinander zu setzen.

Dazu mussten wir unserem LSTM (Long Short Term Memory) Modell zuerst beibringen, wie sich der SMI bis zum heutigen Zeitpunkt verhalten hat. Das LSTM Modell basiert auf dem bereits basierenden Prinzip der RNN (Recurrent Neural Networks) und ist eine Weiterverarbeitung davon. Dieses Update ermöglicht unserem Modell basierend auf dem letzten vorhergesagten Element sowie dem existierenden Datenpunkt den nächsten Datenpunkt vorherzusagen. Um dieses Modell zu trainieren, muss es die Möglichkeit haben, sich selbst zu überprüfen und zu sehen, ob ihre Vorhersage auch stimmt. Abbildung 9 zeigt das Training des LSTM Modells mit den bestehenden Werten des SMI. Der Code, um dieses Modell zu erstellen befindet sich auf GitHub ([Link](https://github.com/TobiasFatzer/STAT_Abgabe/blob/master/ModelCreationOfSMI.py)).

<img src="images\SMI_ML_Prediction.png"/>
 
Abbildung 9 - Trainierte Vorhersage des LSTM Models

Um diese Kalkulationen zu ermöglichen, mussten wir weitere Libraries importieren. Die erste Library, «numpy», wird verwendet, um die Effizienz der Kalkulationen mit Arrays zu erhöhen, da die Datenmodelle in Arrays gespeichert werden. «sklearn» wird verwendet, um eine Skalierte Datenmenge aus der vorgegebenen Datenmenge zu erstellen. «keras» ist eine Library, die für das Erstellen, Verarbeiten und Auswerten der LSTM Modellen zuständig ist. «yfinance» wird verwendet, um alle aktuellen Daten des SMI zu ziehen.

    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import load_model
    import yfinance as yf 

Nach den erfolgten Imports werden die Daten für den in Abbildung 9 gezeigten Graphen erstellt. Der folgende Code zeigt als erstes das Abrufen der Börsendaten von Yahoo mittels der oben genannten «yfinance» Library. Es werden alle Daten zwischen dem 01. Januar 2000 und dem 01. März 2020 abgerufen. Diese werden darauf skaliert, um dem Modell Daten zu geben, welche es auch verstehen kann. Darauf werden die Testdaten in das LSTM Modell abgefüllt und die erste Vorhersage wird vorgenommen und darauf in einem Graphen wie in Abbildung 9 gezeigt dargestellt.
    
    prediction_days = 60
    data = pd.DataFrame(yf.download(tickers=['^SSMI'], start='2000-01-01', end='2020-03-01'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    model = load_model('stock_prediction.smi')
    
    model.summary()
    '''Test Model Accuarcy on Existing Data'''
    
    # Load Test data
    
    test_start = dt.datetime(2020, 4, 1)
    test_end = dt.datetime(2022, 3, 1)
    
    test_data = pd.DataFrame(yf.download(tickers=['^SSMI'], start=test_start, end=test_end))
    actual_prices = test_data['Close'].values
    
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    
    # Make Predictions on test data
    x_test = []
    
    for x in range(prediction_days, len(model_inputs) + 1):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)
    
    x_input = np.array(test_data['Close'][413:]).reshape(1, -1)
    
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    
    lst_output = []
    n_steps = 60
    i = 0
    
    while (i < 30):
        if (len(temp_input) > 60):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1
    
    
    plt.plot(actual_prices, color='black', label='Actual SMI Price')
    plt.plot(prediction_prices, color='green', label='Predicted SMi Price')
    plt.title('SMI Share Price')
    plt.xlabel('Time')
    plt.ylabel('SMI Share Price')
    plt.legend()
    plt.show()
    
    lst_output = lst_output[::-1]
    value_predictions = scaler.inverse_transform(lst_output)
    value_predictions = np.array(value_predictions)
    
    Darauf wird das Model verwendet, um nun zukünftige Vorhersagen vorzunehmen. Dazu wird der folgende Code verwendet.
    
    prediction_dataframe = pd.DataFrame(absolut_df.loc['2022-04-01':]['Close'])
    prediction_dataframe['Close'] = value_predictions[:10]
    absolut_df['Close'] = prediction_dataframe['Close']
    
    if absolut_df['Close'].isnull().values.any():
        absolut_df["Close"].fillna(absolut_df['Absolut_mean'], inplace=True)
    
    Nach erstellen der Tabelle mit den möglichen vorhersagen wird noch ein Graph erstellt, der diese Vorhersagen sowie die Minimale- und Maximale Standartabweichung aufzeigt. Der folgende Code zeigt die Erstellung dieses Graphen.
    plt.plot(absolut_df['Close'], color='#42bbf7', label='Prediction')
    plt.plot(absolut_df['Absolut_Max_Prediction_Value'], color='red', label='Max Potential Value')
    plt.plot(absolut_df['Absolut_mean'], color='green', label='Mean')
    plt.plot(absolut_df['Absolut_Min_Prediction_Value'], color='orange', label='Min Potential Value')
    plt.title('SMI Predictions')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel('Share Price')
    plt.legend()
    plt.show()

Darauf wird das Modell verwendet, um nun zukünftige Vorhersagen vorzunehmen. Dazu wird der folgende Code verwendet:

    prediction_dataframe = pd.DataFrame(absolut_df.loc['2022-04-01':]['Close'])
    prediction_dataframe['Close'] = value_predictions[:10]
    absolut_df['Close'] = prediction_dataframe['Close']

    if absolut_df['Close'].isnull().values.any():
    absolut_df["Close"].fillna(absolut_df['Absolut_mean'], inplace=True)

Nach dem Erstellen der Tabelle mit den möglichen Vorhersagen wird noch ein Graph erstellt, der diese Vorhersagen sowie die minimale- und maximale Standartabweichung aufzeigt. Der folgende Code zeigt die Erstellung dieses Graphen und die darauffolgende Abbildung 10 den gezeichneten Graphen.

    plt.plot(absolut_df['Close'], color='#42bbf7', label='Prediction')
    plt.plot(absolut_df['Absolut_Max_Prediction_Value'], color='red', label='Max Potential Value')
    plt.plot(absolut_df['Absolut_mean'], color='green', label='Mean')
    plt.plot(absolut_df['Absolut_Min_Prediction_Value'], color='orange', label='Min Potential Value')
    plt.title('SMI Predictions')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel('Share Price')
    plt.legend()
    plt.show()

<img src="images\ML_LSTM_Prediction.png" alt="Simulation des SMIs mittels unserem LSTM Model" title="Simulation des SMIs mittels unserem LSTM Model"/>

 Abbildung 10 - Simulation des SMIs mittels unserem LSTM Model

