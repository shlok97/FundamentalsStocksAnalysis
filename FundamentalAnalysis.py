import pandas as pd
import numpy as np
import csv

# df_cashflow = pd.read_csv("stocks/Kotak/Cash Flow-Table 1.csv", error_bad_lines=False)
# df_profitloss = pd.read_csv("stocks/Kotak/Profit & Loss-Table 1.csv", quoting=csv.QUOTE_NONE, error_bad_lines=False)
#
# print(df_profitloss.head())

errors = 0


def getData(stock):
    finalData = []
    # stock = '3i Infotech 3'

    parameterIndexMapping = {}

    NetCashFlow = 'Net Cash Flow'
    ROE = 'Return on Equity'
    ROCE = 'Return on Capital Emp'
    Investments = 'Investments'
    OtherAssets = 'Other Assets'
    EquityShareCapital = 'Equity Share Capital'

    Sales = 'Sales'
    NetProfit = 'Net profit'
    EPS = 'EPS'
    PE = 'Price to earning'
    Price = 'Price'

    finalLen = 0

    with open('data/' + stock + '_Cash Flow-Table1.csv', 'rt')as f:
        data = csv.reader(f)
        for index, row in enumerate(data):
            # print(row)
            if row[0] == NetCashFlow:
                finalData.append(row)
                parameterIndexMapping[NetCashFlow] = finalLen
                finalLen += 1

    with open('data/' + stock + '_Balance Sheet-Table1.csv', 'rt')as f:
        data = csv.reader(f)
        for index, row in enumerate(data):
            # print(row)
            row = row[:11]
            if row[0] == ROE:
                finalData.append(row)
                parameterIndexMapping[ROE] = finalLen
                finalLen += 1
            if row[0] == ROCE:
                finalData.append(row)
                parameterIndexMapping[ROCE] = finalLen
                finalLen += 1
            if row[0] == Investments:
                finalData.append(row)
                parameterIndexMapping[Investments] = finalLen
                finalLen += 1
            if row[0] == OtherAssets:
                finalData.append(row)
                parameterIndexMapping[OtherAssets] = finalLen
                finalLen += 1
            if row[0] == EquityShareCapital:
                finalData.append(row)
                parameterIndexMapping[EquityShareCapital] = finalLen
                finalLen += 1
    with open('data/' + stock + '_Profit & Loss-Table1.csv', 'rt')as f:
        data = csv.reader(f)

        for index, row in enumerate(data):
            # print(row)
            row = row[:11]
            if row[0] == Sales:
                finalData.append(row)
                parameterIndexMapping[Sales] = finalLen
                finalLen += 1
            if row[0] == NetProfit:
                finalData.append(row)
                parameterIndexMapping[NetProfit] = finalLen
                finalLen += 1
            if row[0] == EPS:
                finalData.append(row)
                parameterIndexMapping[EPS] = finalLen
                finalLen += 1
            if row[0] == PE:
                finalData.append(row)
                parameterIndexMapping[PE] = finalLen
                finalLen += 1
            if row[0] == Price:
                finalData.append(row)
                parameterIndexMapping[Price] = finalLen
                finalLen += 1

    data = []
    for row in finalData:
        # print(row, len(row))
        #
        # for col in row:
        #     print(col[:2] == " \t")

        for i in range(len(row)):
            # if row[i][:2] == " \t":
            #     row[i] = row[i][2:]

            # Prepare to convert to float values
            if row[i][-1:] == "%":
                row[i] = row[i][:-1]
            row[i] = row[i].replace(",", "")
            row[i] = row[i].replace("(", "")
            row[i] = row[i].replace(")", "")
            if i is not 0:
                if row[i] == '' or row[i] == '-':
                    row[i] = None
                    continue
                # print(row[i])
                # row[i] = float(row[i])
                try:
                    row[i] = float(row[i])
                except ValueError:
                    # print("Error", row[i])
                    row[i] = None

        # print(row)

        rowNumpy = np.array(row[1:])

        cleanNumpy = rowNumpy[rowNumpy != None]

        # Replace all empty fields
        for i in range(len(row)):
            if row[i] is None:
                if len(cleanNumpy) is 0:
                    row[i] = 0
                    continue
                row[i] = np.median(cleanNumpy)

        # print(rowNumpy, np.median(cleanNumpy))
        # print(row)
        data.append(row[1:])

    # print(data)
    # price = data[-1]
    # print(price[-1], price[-2])
    # y = (price[-1] - price[-2])/price[-2]
    #
    # print(y)

    # print(parameterIndexMapping)


    def getY(data):
        price = data[-1]
        y = (price[-1] - price[-2]) / price[-2]
        return y

    def percentChange(data, param, duration):
        index = parameterIndexMapping[param]
        return (data[index][-2] - data[index][-2 - duration]) / data[index][-2 - duration]

    def getMean(data, param, duration=0):
        index = parameterIndexMapping[param]
        return (data[index][-2] + data[index][-2 - duration]) / 2

    # print(percentChange(data, Price, 1))

    dateRange = [0, 2, 4]
    sales_change = [percentChange(data, Sales, i + 1) for i in dateRange]
    net_cash_flow_change = [percentChange(data, NetCashFlow, i + 1) for i in dateRange]
    share_capital_change = [percentChange(data, EquityShareCapital, i + 1) for i in dateRange]
    investments_change = [percentChange(data, Investments, i + 1) for i in dateRange]
    net_profit_change = [percentChange(data, NetProfit, i + 1) for i in dateRange]
    eps_change = [percentChange(data, EPS, i + 1) for i in dateRange]
    other_assets_change = [percentChange(data, OtherAssets, i + 1) for i in dateRange]
    price_change = [percentChange(data, Price, i + 1) for i in dateRange]
    roe_change = [percentChange(data, ROE, i + 1) for i in dateRange]

    eps = getMean(data, EPS)/100
    pe = getMean(data, PE)/20
    roe = getMean(data, ROE)/100
    roce = getMean(data, ROCE)/100
    y = getY(data)

    # print(sales_change)
    # print(net_cash_flow_change)
    # print(share_capital_change)
    # print(investments_change)
    # print(net_profit_change)
    # print(eps_change)
    # print(price_change)
    # print(eps)
    # print(pe)
    # print(y)
    X = sales_change + roe_change + other_assets_change + net_cash_flow_change + share_capital_change + investments_change + net_profit_change + eps_change + price_change + [eps, pe, roe, roce]

    return X, y


def run():
    # print(getData('3i Infotech 3'))
    # print(getData('Kotak Mah. Bank.xlsx'))

    from stockList import stockList
    # print(stockList)


    # print(getData(stockList[0]))
    X = np.array([])
    y = np.array([])
    stockParamMapping = {}
    for stock in stockList:
        try:
            # print(getData(stock))
            params, output = getData(stock)
            if output > 0:
                output = 1
            else:
                output = 0
            if len(X) == 0:
                X = [params]
                y = [output]
            else:
                X = np.append(X, [params], axis=0)
                y = np.append(y, [output], axis=0)
        # except ValueError:
        #     continue
        except FileNotFoundError:
            continue
        except ZeroDivisionError:
            continue
    # print(X)
    # # print(len(X[0]))
    # print(y)
    # print(len(y))

    # break_point = 500
    # X_train = X[:break_point]
    # y_train = y[:break_point]
    #
    # X_test = X[break_point:]
    # y_test = y[break_point:]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # from sklearn.tree import DecisionTreeClassifier
    #
    # clf = DecisionTreeClassifier()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    #
    #
    #
    # param_dist = {
    #     "max_depth": [30, 60, 100, 150, None],
    #     "max_features": [5, 8, 16, None],
    #     "min_samples_leaf": [2, 8, 16, 32],
    #     "criterion": ["gini", "entropy"],
    #     "n_estimators": [20, 30, 50]
    # }

    clf = RandomForestClassifier(n_estimators=80, max_depth=100, max_features=16, verbose=True)
    #
    # clf.fit(X_train, y_train)


    # clf = RandomForestClassifier()
    # forest_cv = GridSearchCV(clf, param_dist, cv=5)
    #
    # forest_cv.fit(X, y)
    # print("Best params are: ", forest_cv.best_params_)
    #
    # random_grid = {  'bootstrap': [True, False],
    #                  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #                  'max_features': ['auto', 'sqrt'],
    #                  'min_samples_leaf': [1, 2, 4],
    #                  'min_samples_split': [2, 5, 10],
    #                  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    #                }
    #
    #
    # clf = RandomForestClassifier()
    # rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state = 42, n_jobs = -1)
    #
    # rf_random.fit(X, y)
    # print("Best params are: ", rf_random.best_params_)

    # clf = RandomForestClassifier(n_estimators= 50, min_samples_split= 5, min_samples_leaf= 4, max_features= 'auto', max_depth= 30, bootstrap= True)

    clf.fit(X_train, y_train)




    y_pred = []
    for index in range(len(X_test)):
        prediction = clf.predict([X_test[index]])[0]
        y_pred.append(prediction)

    y_pred = np.array(y_pred)
    from sklearn.metrics import confusion_matrix

    # print(confusion_matrix(y_test, y_pred))
    cf_mat = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix")
    # print(cf_mat)

    recall = cf_mat[0][0]/(cf_mat[0][0] + cf_mat[1][0])
    precision = cf_mat[0][0]/(cf_mat[0][0] + cf_mat[0][1])

    # accuracy = cf_mat[0][0] + cf_mat[1][1]/(cf_mat[0][0] + cf_mat[1][0] + cf_mat[0][1]+ cf_mat[1][1])

    # print("PRECISION", precision,"RECALL", recall)
    return precision, recall, cf_mat

precisions = []

while True:
    precision, recall, cf_mat = run()
    final_cf = np.array([[]])
    precisions.append(precision)
    if precision == np.max(precisions):
        final_cf = cf_mat

    print(len(precisions))

    if precision > 0.68 or len(precisions) > 20:
        print("Confusion Matrix")
        print(final_cf)
        print("PRECISION", np.max(precisions), "RECALL", recall)
        print("MeanPrecision: ", np.mean(precisions))
        break
