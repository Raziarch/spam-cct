import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
from matplotlib.ticker import FormatStrFormatter 
from io import BytesIO

def make_dates(n_dates, lag):
    today = datetime.date.today()
    today_datetime = datetime.datetime(day=today.day, month=today.month, year=today.year)
    shift4 = today_datetime - datetime.timedelta(days=4)
    shift4_weekday = shift4.weekday()
    idx = np.arange(n_dates)
    
    f = lambda x: shift4 - datetime.timedelta(days=int(shift4_weekday + 7*(lag+x)))
    
    dates = np.array(list(map(f, idx)))
    dates = dates[::-1]
    return dates

def get_data_dict(dates, score, df, tstmp, paths):
    week_interval = datetime.timedelta(days=7)
    n_apps_list = []
    accuracy_list = []
    n_fraud_list = []
    precision_list = []
    recall_list = []
    caught_list = []
    missed_list = []
    
    #df = df[df['fraud_status'].isin([4, 5, 6])]

    for date in dates:
        start = date
        end = date + week_interval
        mask = ((tstmp >= start) & (tstmp < end))
        
        df_mask = df[mask]
        confirmed_fraud = df_mask['confirmed_fraud']
        marked_fraud = (score >= 0.3)
        
        n_fraud = confirmed_fraud.sum()
        n_apps = df_mask.shape[0]
        
        tp = (confirmed_fraud & marked_fraud).sum()
        tn = ((~confirmed_fraud) & (~marked_fraud)).sum()
        fp = ((~confirmed_fraud) & (marked_fraud)).sum()
        fn = ((confirmed_fraud) & (~marked_fraud)).sum()
        
        if n_apps == 0:
            accuracy = 1
        else:
            accuracy = float(tp + tn)/n_apps
        
        precision_den = tp + fp
        if precision_den == 0:
            precision = 1
        else:
            precision = float(tp)/precision_den
            
        recall_den = tp + fn
        if recall_den == 0:
            recall = 1
        else:
            recall = float(tp)/recall_den
        
        accuracy_list.append(accuracy)
        n_fraud_list.append(n_fraud)
        precision_list.append(precision)
        recall_list.append(recall)
        n_apps_list.append(n_apps)
        caught_list.append(tp)
        missed_list.append(fn)
    
    data_dict = {}
    data_dict['n_entries'] = len(dates)
    data_dict['accuracy'] = accuracy_list
    data_dict['n_apps'] = n_apps_list
    data_dict['n_fraud'] = n_fraud_list
    data_dict['precision'] = precision_list
    data_dict['recall'] = recall_list
    data_dict['caught'] = caught_list
    data_dict['missed'] = missed_list
    data_dict['dates'] = dates
    data_dict['start'] = dates[-1].strftime('%m/%d')
    data_dict['end'] = (dates[-1] + week_interval).strftime('%m/%d')
    data_dict['start_full'] = dates[-1].strftime('%Y%m%d')
    data_dict['end_full'] = (dates[-1] + week_interval).strftime('%Y%m%d')
    data_dict['dates_str'] = list(map(lambda x: x.strftime('%m/%d'), dates))
    data_dict['x_axis'] = np.arange(data_dict['n_entries'])
    data_dict['filename'] = 'spamfilter_report_{}_{}.html'.format(data_dict['start_full'],data_dict['end_full'])
    data_dict['path'] = os.path.join(paths['LOGDIR'], data_dict['filename'])
    return data_dict

def viz_report(data_dict, paths, keep_count):

    fontsize_suptitle = 24
    fontsize_heading = 20
    fontsize_text = 28
    fontsize_title = 16
    fontsize_axis = 14
    fontsize_xlabel = 10
    fontsize_ylabel = 8
    fontsize_legend = 8
    rot_angle = 30
    
    if keep_count:
        plt.figure(figsize=(10,7))
    else:
        plt.figure(figsize=(10,3))
    plt.suptitle('Spam Filter Report: {} - {}'.format(data_dict['start'], data_dict['end']), 
                 fontsize=fontsize_suptitle)
    
    if keep_count:
        plt.subplot(231)
        plt.text(0.5,0.9,'Fraud Count:', va='center', ha='center', fontsize=fontsize_heading)
        plt.text(0.5,0.70,'{:,}'.format(data_dict['n_fraud'][-1]), va='center', ha='center', fontsize=fontsize_text)
        plt.text(0.5,0.4,'Accuracy:', va='center', ha='center', fontsize=fontsize_heading)
        plt.text(0.5,0.20,'{:.1%}'.format(data_dict['accuracy'][-1]), va='center', ha='center', fontsize=fontsize_text)
        plt.axis('off')
    
    if keep_count:
        ax2_subplot = 232
    else:
        ax2_subplot = 131
    ax2 = plt.subplot(ax2_subplot)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('% 1.3f'))
    plt.title('Historical Accuracy', fontsize=fontsize_title)
    plt.plot(data_dict['x_axis'], data_dict['accuracy'])
    plt.scatter(data_dict['x_axis'], data_dict['accuracy'])
    plt.xticks(data_dict['x_axis'], data_dict['dates_str'], fontsize=fontsize_xlabel, rotation=rot_angle)
    plt.xlabel('Week', fontsize=fontsize_axis)
    plt.yticks(fontsize=fontsize_ylabel)
    plt.subplots_adjust(top=0.87)

    if keep_count:
        plt.subplot(233)
        plt.title('Historical Fraud Count', fontsize=fontsize_title)
        plt.xlabel('Week', fontsize=fontsize_axis)
        plt.bar(data_dict['x_axis'], data_dict['n_fraud'])
        plt.xticks(data_dict['x_axis'], data_dict['dates_str'], fontsize=fontsize_xlabel, rotation=rot_angle)
        plt.yticks(fontsize=fontsize_ylabel)

    if keep_count:
        plt.subplot(234)
        y_max = max(max(data_dict['caught']), max(data_dict['missed']))*1.3
        width = 0.35
        plt.xlabel('Week', fontsize=fontsize_axis)
        plt.yticks(fontsize=fontsize_ylabel)
        plt.bar(data_dict['x_axis'] - width/2, data_dict['caught'], width, label='Caught Fraud')
        plt.bar(data_dict['x_axis'] + width/2, data_dict['missed'], width, label='Missed Fraud')
        plt.title('Historical Fraud Metrics', fontsize=fontsize_title)
        plt.xticks(data_dict['x_axis'], data_dict['dates_str'], fontsize=fontsize_xlabel, rotation=rot_angle)
        plt.legend(fontsize=fontsize_legend,)
        plt.ylim((0,y_max))

    if keep_count:
        ax5_subplot = 235
    else:
        ax5_subplot = 132
    ax5 = plt.subplot(ax5_subplot)
    ax5.yaxis.set_major_formatter(FormatStrFormatter('% 1.2f')) 
    plt.plot(data_dict['x_axis'], data_dict['precision'], label='Precision')
    plt.scatter(data_dict['x_axis'], data_dict['precision'])
    plt.xticks(data_dict['x_axis'], data_dict['dates_str'], fontsize=fontsize_xlabel, rotation=rot_angle)
    plt.xlabel('Week', fontsize=fontsize_axis)
    plt.yticks(fontsize=fontsize_ylabel)
    plt.title('Historical Precision', fontsize=fontsize_title)
    plt.subplots_adjust(top=0.87)

    if keep_count:
        ax6_subplot = 236
    else:
        ax6_subplot = 133
    ax6 = plt.subplot(ax6_subplot)
    ax6.yaxis.set_major_formatter(FormatStrFormatter('% 1.2f')) 
    plt.plot(data_dict['x_axis'], data_dict['recall'], label='Recall')
    plt.scatter(data_dict['x_axis'], data_dict['recall'])
    plt.xticks(data_dict['x_axis'], data_dict['dates_str'], fontsize=fontsize_xlabel, rotation=rot_angle)
    plt.xlabel('Week', fontsize=fontsize_axis)
    plt.yticks(fontsize=fontsize_ylabel)
    plt.title('Historical Recall', fontsize=fontsize_title)
    
    if keep_count:
        subplot_top = 0.85
    else:
        subplot_top = 0.75
    plt.subplots_adjust(top=subplot_top)


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.20, hspace=0.45)

    #plt.savefig(data_dict['path'])
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png', bbox_inches = "tight")
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    logo_path = os.path.join(paths['UTILDIR'], 'ccctc_logo.png')
    legend_path = os.path.join(paths['UTILDIR'], 'legend.png')
    encoded_logo = base64.b64encode(open(logo_path, 'rb').read()).decode('utf-8')
    encoded_legend = base64.b64encode(open(legend_path, 'rb').read()).decode('utf-8')
    style = 'display:block; margin-left:auto; margin-right:auto;'
    html = '<!DOCTYPE html>\n<html>\n<body>\n' + '<img src=\'data:image/png;base64,{}\' style=\'{} width:400px\'>\n<br>\n'.format(encoded_logo, style) + '<img src=\'data:image/png;base64,{}\' style=\'{}\'>\n<br>\n'.format(encoded, style) +  '<img src=\'data:image/png;base64,{}\' style=\'{} width:60%\'>\n<br>\n'.format(encoded_legend, style) + '\n</body>\n</html>\n'

    with open(data_dict['path'],'w') as f:
        f.write(html)
    
def make_report(df_train, score, reportopts, paths, log):
    dates = make_dates(reportopts['n_dates'], reportopts['lag'])
    tstmp = pd.to_datetime(df_train['tstmp_submit'])
    data_dict = get_data_dict(dates, score, df_train, tstmp, paths)
    log.info('Report written to: {}'.format(data_dict['path']))
    viz_report(data_dict, paths, reportopts['keep_count'])
