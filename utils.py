import matplotlib.pyplot as plt
import numpy as np


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    save_result_file(sorted_result)

    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        print(name, ':', '%5.3f  ' % score, '   ',)
    print
    return sorted_result


def save_result_file(sorted_result):
    file = open('result.txt', 'w')
    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        file.write(name + ':' + '%5.3f' % score + ';')
    file.close()


def predict_label(preds):
    labels=preds.argsort()[::-1]
    return labels[0]


def change_to_json(sorted_result):
    ziped = []
    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        ziped.append((name, '%5.3f  ' % score))
    import json
    json = json.dumps([{'genre': genre, 'pred': pred} for genre, pred in ziped])
    return json


def print_plot(mean, tags):
    # plot with Genres percentage classification
    colors = ['b','g','c','r','m','k','y','#ff1122','#5511ff','#44ff22']
    fig, ax = plt.subplots()
    index = np.arange(tags.shape[0])
    opacity = 1
    bar_width = 0.2
    print(mean)
    for g in range(0, tags.shape[0]):
        plt.bar(left=index, height=mean, width=bar_width, alpha=opacity, color=colors)

    plt.xlabel('Genres')
    plt.ylabel('Percentage')
    plt.title('Scores by genre')
    plt.xticks(index + bar_width / 2, tags)
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.savefig('genres_prediction.png')
