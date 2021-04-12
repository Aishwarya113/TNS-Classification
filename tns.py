from __future__ import print_function
import numpy as np
import pandas as pd
import os, glob2
import requests, json
from lxml import html
from astropy.time import Time
import pickle
from astropy.table import Table
from astropy.io import ascii
import os
import requests
import json
from datetime import datetime
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy import constants as const
import sys, getopt, argparse
import re
from time import sleep
from astropy.io import fits
from subprocess import call
from lxml import html
import webbrowser as wb
from urllib.error import HTTPError
import xlsxwriter

global TOKEN, BASEURL
GETTOKEN = ''      # Fritz API Key, input your TOKEN from Fritz
BASEURL = 'https://fritz.science/'                     # Fritz base url

API_KEY = "54916f1700966b3bd325fc1189763d86512bda1d"     # TNS API Key

# TNS URLs for real uploads
TNS_BASE_URL = "https://www.wis-tns.org/api/"    #Access TNS using https://www.wis-tns.org/
upload_url = "https://www.wis-tns.org/api/file-upload"
report_url = "https://www.wis-tns.org/api/bulk-report"
reply_url = "https://www.wis-tns.org/api/bulk-report-reply"

# SANDBOX URLs for TNS upload trials
SAND_TNS_BASE_URL = "https://sandbox.wis-tns.org/api/"  #Access Sandbox using https://sandbox.wis-tns.org/
SAND_upload_url = "https://sandbox.wis-tns.org/api/file-upload"
SAND_report_url = "https://sandbox.wis-tns.org/api/bulk-report"
SAND_reply_url = "https://sandbox.wis-tns.org/api/bulk-report-reply"


def api(method, endpoint, data=None):
    ''' Info : Basic API query, takes input the method (eg. GET, POST, etc.), the endpoint (i.e. API url)
               and additional data for filtering
        Returns : response in json format
        CAUTION! : If the query doesn't go through, try putting the 'data' input in 'data' or 'params'
                    argument in requests.request call
    '''
    headers = {'Authorization': f'token {GETTOKEN}'}
    response = requests.request(method, endpoint, json=data, headers=headers)
    return response.json()



def get_source_api(ztfname):
    ''' Info : Query a single source, takes input ZTF name
        Returns : all basic data of that source (excludes photometry and spectra,
                  includes redshift, classification, comments, etc.)
    '''
    url = BASEURL+'api/sources/'+ztfname+'?includeComments=true'
    response = api('GET',url)
    return response['data']



def get_group_ids(groupnames=['Redshift Completeness Factor', 'Census of the Local Universe Caltech']):
    ''' Info : Query group ids of groups specified
        Input : Name or names of groups in an array []
        Returns : List of group  names and their group ids
    '''

    url = BASEURL+'api/groups'
    headers = {'Authorization': f'token {GETTOKEN}'}
    groupnames = np.atleast_1d(groupnames)
    grpids = []
    for grpname in groupnames:
        response = requests.request('GET',url,params={'name':grpname}, headers=headers).json()
        answer = str(grpname)+' = '+str(response['data'][0]['id'])
        grpids.append(answer)

    return grpids



def get_number_of_sources(group_id, date):
    ''' Info : Query number of sources saved in a group after a certain date
        Input : group id, date [yyyy-mm-dd]
        Returns : Number of sources saved after a given date to the specified group
    '''

    url = BASEURL+'api/sources?saveSummary=true&group_ids='+group_id+'&savedAfter='+date+'T00:00:00.000001'
    response = api('GET',url)
    return len(response['data']['sources'])



def get_group_sources(group_id, date):
    ''' Info : Query all sources saved in a group after a certain date
        Input : group id, date [yyyy-mm-dd]
        Returns : List of jsons of all sources in group(s)
        Comment : Takes a little time based on the date
    '''

    sources = []

    for i in range (get_number_of_sources(group_id, date)):

        url = BASEURL+'api/sources?saveSummary=true&group_ids='+group_id+'&savedAfter='+date+'T00:00:00.000001'
        response = api('GET',url)
        ztfname = response['data']['sources'][i]['obj_id']
        sources.append(ztfname)

    return sources



def get_total_number_of_sources(group_id):
    ''' Info : Query total number of sources saved in a group
        Input : group id
        Returns : Total number of sources saved in a group
    '''

    url = BASEURL+'api/sources?saveSummary=true&group_ids='+group_id
    response = api('GET',url)
    return len(response['data']['sources'])


def get_all_group_sources(group_id):
    ''' Info : Query all sources saved in a group
        Input : group id
        Returns : List of jsons of all sources in group(s)
        Comment : Takes a long time
    '''

    sources = []

    for i in range (get_number_of_sources(group_id)):

        url = BASEURL+'api/sources?saveSummary=true&group_ids='+group_id
        response = api('GET',url)
        ztfname = response['data']['sources'][i]['obj_id']
        sources.append(ztfname)

    return sources


def get_IAUname(ztfname):

    ''' Info : Query the TNS name for any source
        Input : ZTFname
        Returns : ATname
    '''

    url = BASEURL+'api/alerts/ztf/'+ztfname+'/aux'
    response = api('GET',url)
    return response["data"]["cross_matches"]["TNS"]


def get_classification(ztfname):

    ''' Info : Query the classification and classification date for any source
        Input : ZTFname
        Returns : Classification and Classification date
        Comment : You need to choose the classification if there are multiple classifications
    '''

    url = BASEURL+'api/sources/'+ztfname+'/classifications'
    response = api('GET',url)
    output = response['data']

    if (len(output)< 1):
        classification = "No Classification found"
        classification_date = "None"

    if (len(output)==1):

        classification = response['data'][0]['classification']
        classification_date = response['data'][0]['created_at'].split('T')[0]

    if (len(output) > 1):

        classification = []
        classification_date = []

        for i in range (len(output)):

            classify = response['data'][i]['classification']
            classify_date = response['data'][i]['created_at']

            classification.append(classify)
            classification_date.append(classify_date)

        for i in range (len(classification)):

            print ((i+1),")", "Classification: ", classification[i],  "\t Classification date:", classification_date[i].split('T')[0])

        user_input = input("Choose classification: ")

        classification = classification[int(user_input)-1]
        classification_date = classification_date[int(user_input)-1].split('T')[0]

    return classification, classification_date


def get_redshift(ztfname):

    ''' Info : Query the redshift for any source
        Input : ZTFname
        Returns : redshift
    '''

    url = BASEURL+'api/sources/'+ztfname
    response = api('GET',url)

    redshift = response['data']['redshift']

    if (redshift == None):
        redshift = "No redshift found"

    return redshift


def get_TNS_information(ztfname):

    url = BASEURL+'api/sources/'+ztfname
    response = api('GET',url)

    IAU = get_IAUname(ztfname)

    if not IAU:
        IAU = "Not reported to TNS"

    else:
        IAU = IAU[0]['name']

    clas = get_classification(ztfname)

    if clas[1] == 'None':
        clas = "Not classified yet"

    else:
        clas = ('Classification: '+str(clas[0])+','+' Classification date: '+str(clas[1]))

    redshift = get_redshift(ztfname)

    if redshift == None:
        redshift = "No redshift found"

    else:
        redshift = ('redshift:'+str(redshift))

    return ztfname, IAU, clas, redshift


def convert_to_jd(date):

    d = Time(date, format='fits')
    dat = d.jd
    return dat


def get_spectrum_api(spectrum_id):
    ''' Info : Query all spectra corresponding to a source, takes input ZTF name
        Returns : list of spectrum jsons
    '''
    url = BASEURL+'api/spectrum/'+str(spectrum_id)
    response = api('GET',url)
    return response


def get_all_spectra_len(ztfname):

    url = BASEURL+'api/sources/'+ztfname+'/spectra'
    response = api('GET',url)
    return len(response['data']['spectra'])


def get_all_spectra_id(ztfname):
    ''' Info : Query all spectra corresponding to a source, takes input ZTF name
        Returns : list of spectrum jsons
    '''

    spec_id = []

    for i in range (get_all_spectra_len(ztfname)):

        url = BASEURL+'api/sources/'+ztfname+'/spectra'
        response = api('GET',url)

        specid = response['data']['spectra'][i]['id']
        spec_id.append(specid)

    return spec_id


def get_required_spectrum_id(ztfname):

    flag = 0

    spec = (get_all_spectra_len(ztfname))

    name = []
    date = []

    if spec == 0:

        specid = "No Spectra Found"
        flag = 1

    if flag == 0:

        spec_id = get_all_spectra_id(ztfname)

        for s in range (spec):

            url = BASEURL+'api/sources/'+ztfname+'/spectra'
            response = api('GET',url)

            spec_name = response['data']['spectra'][s]['original_file_filename']
            spec_date = response['data']['spectra'][s]['observed_at']

            name.append(spec_name)
            date.append(spec_date.split('T')[0])

        print ("Please choose from the following spectra: \n")

        for i in range (len(name)):
            print ((i+1),")", "spectrum name: ", name[i], "spectrum date:", date[i])

        wb.open(BASEURL+'source/'+ztfname, new=2)

        user_input = input("Choose spectrum to upload: ")

        specid = spec_id[int(user_input)-1]

    return specid



def write_ascii_file(ztfname):

    specid = get_required_spectrum_id(ztfname)

    flag = 0

    if (specid == 'No Spectra Found'):
        spectrum_name = 'No Spectra Found'
        print (spectrum_name)
        flag = 1

    if flag == 0:

        a = get_spectrum_api(specid)

        inst = (a['data']['instrument_name'])
        #print (inst)

        if inst == 'SEDM':

            header = (a['data']['altdata'])
            path = os.getcwd()

            s = (ztfname+'_'+str(header['OBSDATE'])+'_'+str(inst)+'.ascii')

            with open(path+'/data/'+s,'w') as f:
                f.write(a['data']['original_file_string'])
            f.close()

            #print (s,'\n')
            spectrum_name = s


        if inst == 'SPRAT':


            header = (a['data']['altdata'])

            path = os.getcwd()

            s = (ztfname+'_'+str(header['OBSDATE'].split('T')[0])+'_'+str(inst)+'.ascii')

            with open(path+'/data/'+s,'w') as f:
                f.write(a['data']['original_file_string'])
            f.close()

            #print (s,'\n')
            spectrum_name = s


        if inst == 'ALFOSC':

            OBSDATE = a['data']['observed_at'].split('T')[0]
            path = os.getcwd()

            s = (ztfname+'_'+str(OBSDATE)+'_'+str(inst)+'.ascii')

            with open(path+'/data/'+s,'w') as f:
                f.write(a['data']['original_file_string'])
            f.close()

            #print (s,'\n')
            spectrum_name = s


        if inst == 'DBSP':

            wav = (a['data']['wavelengths'])
            flux = (a['data']['fluxes'])
            err = (a['data']['errors'])

            OBSDATE = a['data']['observed_at'].split('T')[0]

            path = os.getcwd()

            s = (ztfname+'_'+str(OBSDATE)+'_'+str(inst)+'.ascii')

            if err == None:

                with open(path+'/data/'+s,'w') as f:

                    for i in range(len(wav)):
                        f.write(str(wav[i])+'\t'+str(flux[i])+'\n')
                f.close()

                #print (s,'\n')
                spectrum_name = s

            else:

                with open(path+'/data/'+s,'w') as f:

                    for i in range(len(wav)):
                        f.write(str(wav[i])+'\t'+str(flux[i])+'\t'+str(err[i])+'\n')
                f.close()

                #print (s,'\n')
                spectrum_name = s


        if inst == 'DIS':

            #obsdate = a['data']['original_file_string'].split('#')[6]
            #a,b = obsdate.split(' ', 1)
            #c,OBSDATE = b.split(' ', 1)
            #OBSDATE = OBSDATE.split('T')[0]

            obsdate = a['data']['observed_at']
            OBSDATE = obsdate.split('T')[0]

            path = os.getcwd()
            path = path+'/data/'

            s = (ztfname+'_'+str(OBSDATE)+'_'+str(inst)+'.ascii')

            a = get_spectrum_api(specid)

            with open(path+s,'w') as f:
                f.write(a['data']['original_file_string'])
            f.close()

            #print (s,'\n')
            spectrum_name = s


        if inst == 'LRIS':

            wav = (a['data']['wavelengths'])
            flux = (a['data']['fluxes'])
            err = (a['data']['errors'])

            OBSDATE = a['data']['observed_at'].split('T')[0]

            path = os.getcwd()

            s = (ztfname+'_'+str(OBSDATE)+'_'+str(inst)+'.ascii')

            if err == None:

                with open(path+'/data/'+s,'w') as f:

                    for i in range(len(wav)):
                        f.write(str(wav[i])+'\t'+str(flux[i])+'\n')
                f.close()

                #print (s,'\n')
                spectrum_name = s

            else:

                with open(path+'/data/'+s,'w') as f:

                    for i in range(len(wav)):
                        f.write(str(wav[i])+'\t'+str(flux[i])+'\t'+str(err[i])+'\n')
                f.close()

                #print (s,'\n')
                spectrum_name = s


        if inst == 'EFOSC2':

            spectrum_name = 'TNS_spectrum'

    return spectrum_name, specid



def APO(specid):

    a = get_spectrum_api(specid)
    inst = (a['data']['instrument_name'])

    obsdate = a['data']['original_file_string'].split('#')[6]
    a,b = obsdate.split(' ', 1)
    c,obsdate = b.split(' ', 1)

    OBSDATE = obsdate

    a = get_spectrum_api(specid)

    exptime = a['data']['original_file_string'].split('#')[9]
    a,b = exptime.split(' ', 1)
    c,EXPTIME = b.split(' ', 1)

    a = get_spectrum_api(specid)

    observers = (a['data']['original_file_string'].split('#')[10])
    a,b = observers.split(' ', 1)
    c,OBSERVERS = b.split(' ', 1)

    a = get_spectrum_api(specid)

    reducers = a['data']['original_file_string'].split('#')[11]
    a,b = reducers.split(' ', 1)
    c,d = b.split(' ', 1)
    REDUCERS,e = d.split('\n', 1)

    return OBSDATE.split(' \n')[0], EXPTIME.split(' \n')[0], OBSERVERS.split(' \n')[0], REDUCERS


def pprint(*args, **kwargs):
    """
    slightly more convenient function instead of print(get_pprint)

    params:
        *args (arguments to pass to get_pprint)
        **kwargs (keyword arguments to pass to get_pprint)
    """
    print(get_pprint(*args, **kwargs))


def post_comment(ztfname, text):


    data = {  "obj_id": ztfname,
              "text": text,
           }

    url = BASEURL+'api/comment'

    response = api('POST', url, data=data)

    return response


def pprint(*args, **kwargs):
    """
    slightly more convenient function instead of print(get_pprint)

    params:
        *args (arguments to pass to get_pprint)
        **kwargs (keyword arguments to pass to get_pprint)
    """
    print(get_pprint(*args, **kwargs))


def get_pprint(item, indent=0, tab=' '*4, maxwidth=float('inf')):
    """
    it's just like 'from pprint import pprint', except instead of
    having dictionaries use hanging indents dependent on the length
    of their key, if the value is a list or dict it prints it indented
    by the current indent plus tab

    params:
        item <di or li> (the thing to be printed)
        indent <int> (the number of times it's been indented so far)
        tab <str> (how an indent is represented)
        maxwidth <int> (maximum characters per line in ouptut)

    returns:
        result <str>
    """
    def get_pprint_di(di, indent, tab=' '*4):
        """
        pprints a dictionary

        params:
            di <dict>
            indent <int> (the number of indents so far)

        returns:
            di_str <str>
        """
        di_str = ''
        for i, (key, item) in enumerate(di.items()):
            di_str += tab*indent
            di_str += repr(key) + ': ' + get_pprint(item, indent, tab)
            if i+1 < len(di):
                # everything until the last item has a trailing comma
                di_str += ',\n'
            else:
                di_str += '\n'
        return di_str

    def get_pprint_li(li, indent, tab=' '*4):
        """
        pprints a list

        params:
            li <list>
            indent <int> (the number of indents so far)

        returns:
            current_result <str>
        """
        li_str = ''
        for i, item in enumerate(li):
            li_str += tab*indent
            pprint(item, indent, tab)
            if i+1 < len(li):
                li_str += ',\n'
            else:
                li_str += '\n'
        return li_str

    result = ''
    if isinstance(item, dict):
        result += '{\n'
        result += get_pprint_di(item, indent+1, tab)
        result += tab*indent + '}'
    elif isinstance(item, list):
        result += '[\n'
        result += get_pprint_li(item, indent+1, tab)
        result += tab*indent + ']'
    else:
        result += repr(item)


    # this gets rid of too-long lines, but only supports space tabs
    lines = result.split('\n')
    for i, line in enumerate(lines):
        while max([len(li) for li in line.split('\n')]) > maxwidth:
            tabs = line[:-len(line.lstrip())]
            if len(tabs) > maxwidth - 8:
                break # giving up
            line = line[:78] + '\\\n' + tabs + 2*tab + line[78:]
            lines[i] = line
    result = '\n'.join(lines)

    return result


def get_number(group_id, date):
    ''' Info : Query number of sources saved in a group after a certain date
        Input : group id, date [yyyy-mm-dd]
        Returns : Number of sources saved after a given date to the specified group
    '''

    url = BASEURL+'api/sources?saveSummary=true&group_ids='+group_id+'&savedAfter='+date+'T00:00:00.000001'
    response = api('GET',url)
    return len(response['data']['sources'])


def get_sources(group_id, date):
    ''' Info : Query all sources saved in a group after a certain date
        Input : group id, date [yyyy-mm-dd]
        Returns : List of jsons of all sources in group(s)
        Comment : Takes a little time based on the date
    '''

    sources = []

    for i in range (get_number(group_id, date)):

        url = BASEURL+'api/sources?saveSummary=true&group_ids='+group_id+'&savedAfter='+date+'T00:00:00.000001'
        response = api('GET',url)
        ztfname = response['data']['sources'][i]['obj_id']
        sources.append(ztfname)

    return sources

def downloadfritzascii(outfile):

    date = '2020-11-06' #Specify the date from which you want to check the saved sources

    path = 'https://fritz.science/api/sources?group_ids=41&saveSummary=true&savedAfter='+date+'T00:00:00.000001'

    response = api('GET',path)

    srcs = []
    dates = []

    listdir = os.getcwd()
    f = open (listdir+'/'+outfile+'.ascii','w')
    f.write('col1'+'\t'+'col2'+'\n')

    for i in range (get_number('41', date)):

        source_name = response['data']['sources'][i]['obj_id']
        saved_date = response['data']['sources'][i]['saved_at']

        srcs.append(source_name)
        dates.append(saved_date.split('T')[0])


    #sorted_dates = sorted(dates, reverse = True)
    #Sources = [x for _,x in sorted(zip(dates,srcs), reverse=True)]

    output = sorted(zip(dates, srcs), reverse=True)

    for i in range (get_number('41', date)):

        f.write(output[i][1]+'\t'+output[i][0]+'\n')

    f.close()

    return


def get_TNS_classification_ID(classification):

    class_ids = {'Afterglow':23, 'AGN':29, 'CV':27, 'Galaxy':30, 'Gap':60, 'Gap I':61, 'Gap II':62, 'ILRT':25, 'Kilonova':70, 'LBV':24,'M dwarf':210, 'Nova':26, 'Novae':26, 'QSO':31, 'SLSN-I':18, 'SLSN-II':19, 'SLSN-R':20, 'SN':1, 'I':2, 'Type I':2, 'I-faint':15, 'I-rapid':16, 'Ia':3, 'Ia-norm':3, 'Ia-91bg':103,'Ia-91T':104, 'Ia-CSM':106, 'Ia-pec':100, 'Ia-SC':102, 'Ia-02cx':105,
                'Ib':4, 'Ib-norm':4, 'Ib-Ca-rich':8, 'Ib-pec':107, 'Ib/c':6, 'SN Ibn':9, 'Ic':5, 'Ic-norm':5, 'Ic-BL':7, 'Ic-pec':108, 'II':10, 'Type II':10, 'II-norm':10,
                'II-pec':110, 'IIb':14, 'IIL':12, 'IIn':13, 'IIn-pec':112, 'IIP':11, 'SN impostor':99, 'Std-spec':50, 'TDE':120,
                'Varstar':28, 'WR':200, 'WR-WC':202, 'WR-WN':201, 'WR-WO':203, 'Other':0}

    #keys = np.array(class_ids.keys())
    for keys in class_ids:
        if (keys == classification):
            classkey = class_ids[keys]
            return classkey


def get_TNS_instrument_ID(classification):

    inst_ids = {'DBSP':1, 'ALFOSC': 41, 'LRIS': 3, 'DIS': 70, 'SEDM': 149, 'SPRAT': 156, 'GMOS': 6, 'Lick-3m': 10, 'LFC': 2, 'TSPEC': 109}

    #keys = np.array(class_ids.keys())
    for keys in inst_ids:
        if (keys == inst):
            instkey = inst_ids[keys]
            return instkey


class TNSClassificationReport:
    def __init__(self):
        self.name = ''
        self.fitsName = ''
        self.asciiName = ''
        self.classifierName = ''
        self.classificationID = ''
        self.redshift = ''
        self.classificationComments = ''
        self.obsDate = ''
        self.instrumentID = ''
        self.expTime = ''
        self.observers = ''
        self.reducers = ''
        self.specTypeID = ''
        self.spectrumComments = ''
        self.groupID = ''
        self.spec_proprietary_period_value = ''
        self.spec_proprietary_period_units = ''


    def fill(self):
        spectrumdict = {
            'obsdate': self.obsDate,
            'instrumentid': self.instrumentID,
            'exptime': self.expTime,
            'observer': self.observers,
            'reducer': self.reducers,
            'spectypeid': self.specTypeID,
            'ascii_file': self.asciiName,
            'fits_file': self.fitsName,
            'remarks': self.spectrumComments,
            'spec_proprietary_period' : self.spec_proprietary_period_value}

        classificationdict =  {
            'classification_report': {
                '0': {
                    'name': self.name,
                    'classifier': self.classifierName,
                    'objtypeid': self.classificationID,
                    'redshift': self.redshift,
                    'groupid': self.groupID,
                    'remarks': self.classificationComments,
                    'spectra': {
                        'spectra-group': {
                            '0': spectrumdict
                        }
                    }
                }
            }
        }

        return classificationdict

    def as_json(self):
        return json.dumps(self.fill())


    def classificationJson(self):
        return json.dumps(self.classificationDict())


def upload_to_TNS(filename, base_url = upload_url, api_key = API_KEY, filetype='ascii'):  #change "base_url = upload_url" to "base_url = SAND_upload_url" to use Sandbox
    """
    uploads a file to TNS and returns the response json
    """
    url = base_url
    data = {'api_key' : api_key}

    if filetype is 'ascii':
        files = [('files[]', (filename, open(filename), 'text/plain'))]

    elif filetype is 'fits':
        files = [('files[0]', (filename, open(filename, 'rb'),
                               'application/fits'))]

    if filename:
        response = requests.post(url, data=data, files=files)
        try:
            return response.json()
        except:
            print(url, data, files, response.content, sep='\n')
            return False
    else:
        return {}



def tns_classify(classificationReport, base_url=report_url, api_key=API_KEY):   #change "base_url = report_url" to "base_url = SAND_report_url" to use Sandbox
    """
    submits classification report to TNS and returns the response json
    """
    url = base_url
    data = {'api_key' : api_key, 'data' : classificationReport.classificationJson()}
    response = requests.post(url, data=data).json()
    if not response:
        return False

    res_code = response['id_code']
    report_id = response['data']['report_id']
    print("ID:", report_id)
    print(res_code, response['id_message'], "reporting finished")
    if res_code == 200:
        return report_id
    else:
        print("Result reporting didn't work")
        pprint(response)
        print("re-submit classification, but don't re-upload files")
        return False

def tns_feedback(report_id):
    data = {'api_key': API_KEY, 'report_id': report_id}  
    response = requests.post(reply_url, data=data).json()       #change "reply_url" to "SAND_reply_url" to use Sandbox
    feedback_code = response['id_code']
    print(feedback_code, response['id_message'], "feedback finished")
    if feedback_code == 200:
        return True
    elif feedback_code == 404:
        print("Waiting and retrying...")
        sleep(2)
        try:
            return tns_feedback(report_id)
        except KeyboardInterrupt:
            return False
    elif feedback_code == 400:
        print(response)
        return False
    else:
        # error receiving the feedback from TNS about the upload
        print("Something went wrong with the feedback, but the report may",
              "still have been fine?")
        return False



downloadfritzascii('RCF_sources') #download the updated list of sources saved to RCF in descending order

f = ascii.read("RCF_sources.ascii") #ascii file containing the names of sources and their saved dates
sources = f['col1']

#sources = ['ZTF21aanyyde', 'ZTF21aanyfqu', 'ZTF21aantsla', 'ZTF21aanfuuk', 'ZTF21aamnngb', 'ZTF21aaljjvt'] #Enter the list of sources you want to report


for source in sources:

    flag = 0
    ztfname = source

    for i in range (len(get_source_api(source)['comments'])):

        comment = get_source_api(source)['comments'][i]['text']

        if comment == 'Uploaded to TNS':
            flag = 1

    if flag == 0:

        info = get_TNS_information(ztfname)

        if info[2] == 'Not classified yet':         #Check if classified
            flag =1
            continue


        a,b = (info[2]).split(',', 1)               #This is just to get the classification date
        c,d = b.split(':', 1)
        e,class_date = d.split(' ', 1)


        k,l = (info[2]).split(':', 1)               #This is just to get the classification
        m,n = l.split(',', 1)
        o,classify = m.split(' ', 1)

        #print (classify, class_date)

        print (info)
        print ("Do you want to proceed with the report?")

        user_input = input("y/n: ")
        print('\n')

        if user_input == 'y':

            spectrum_info = write_ascii_file(ztfname) #returns "spectrum_name"

            spectrum_name = spectrum_info[0]

            if spectrum_name == 'No Spectra Found':
                flag = 1
                continue

            if flag == 0:

                path = os.getcwd()

                specfile = (path+'/data/'+spectrum_name)

                files = specfile

                specid = spectrum_info[1]

                a = get_spectrum_api(specid)

                inst = (a['data']['instrument_name'])


                if inst == 'SEDM':

                    classifiers = 'A. Dahiwale, C. Fremling(Caltech) on behalf of the Zwicky Transient Facility (ZTF)'### Change accordingly
                    source_group = 48 ### Require source group id from drop down list, 0 is for None
                    spectypes = np.array(['object','host','sky','arcs','synthetic'])

                    #proprietary_period = int(input("Proprietary period in years:", x)
                    proprietary_period = '0'
                    proprietary_units = "years"
                    spec_comments =''
                    classification_comments = ''
                    spectype='object'
                    spectype_id = ['object', 'host', 'sky', 'arcs', 'synthetic'].index(spectype) + 1

                    header = (a['data']['altdata'])
                    obsdate = str((header['UTC']).split('T')[0])+' '+str((header['UTC']).split('T')[1])

                    classificationReport = TNSClassificationReport()
                    classificationReport.name = get_IAUname(ztfname)[0]['name'][3:]
                    classificationReport.fitsName = ''
                    classificationReport.asciiName = spectrum_name
                    classificationReport.classifierName = classifiers
                    classificationReport.classificationID = get_TNS_classification_ID(classify)
                    classificationReport.redshift = get_redshift(ztfname)
                    classificationReport.classificationComments = classification_comments
                    classificationReport.obsDate = obsdate
                    classificationReport.instrumentID = get_TNS_instrument_ID(inst)
                    classificationReport.expTime = (header['EXPTIME'])
                    classificationReport.observers = 'SEDmRobot'
                    classificationReport.reducers = (header['REDUCER'])
                    classificationReport.specTypeID = spectype_id
                    classificationReport.spectrumComments = spec_comments
                    classificationReport.groupID = source_group
                    classificationReport.spec_proprietary_period_value = proprietary_period
                    classificationReport.spec_proprietary_period_units = proprietary_units

                    pprint(classificationReport.fill(), tab='  ')
                    proceed = input("\nProceed with classification and upload? ([y]/n) : ")
                    if proceed == 'y' and not proceed.strip() == '':

                        # ASCII FILE UPLOAD
                        print ("\n")
                        response = upload_to_TNS(files)
                        print (response)

                        if not response:
                            print("File upload didn't work")
                            print(response)
                            #return False

                        print(response['id_code'], response['id_message'],
                              "\nSuccessfully uploaded ascii spectrum")
                        #classificationReport.asciiName = response['data'][-1]

                        report_id = tns_classify(classificationReport)
                        post_comment(ztfname, 'Uploaded to TNS')
                        tns_feedback(report_id)


                if inst == 'SPRAT':

                    classifiers = 'D. A. Perley, K. Taggart (LJMU), A. Dahiwale, C. Fremling (Caltech) on behalf of the Zwicky Transient Facility (ZTF)'### Change accordingly
                    source_group = 48 ### Require source group id from drop down list, 0 is for None
                    spectypes = np.array(['object','host','sky','arcs','synthetic'])

                    #proprietary_period = int(input("Proprietary period in years:", x)
                    proprietary_period = '0'
                    proprietary_units = "years"
                    spec_comments =''
                    classification_comments = ''
                    spectype='object'
                    spectype_id = ['object', 'host', 'sky', 'arcs', 'synthetic'].index(spectype) + 1

                    header = (a['data']['altdata'])
                    obsdate = str(header['OBSDATE'].split('T')[0])+' '+str(header['OBSDATE'].split('T')[1])

                    classificationReport = TNSClassificationReport()
                    classificationReport.name = get_IAUname(ztfname)[0]['name'][3:]
                    classificationReport.fitsName = ''
                    classificationReport.asciiName = spectrum_name
                    classificationReport.classifierName = classifiers
                    classificationReport.classificationID = get_TNS_classification_ID(classify)
                    classificationReport.redshift = get_redshift(ztfname)
                    classificationReport.classificationComments = classification_comments
                    classificationReport.obsDate = obsdate
                    classificationReport.instrumentID = get_TNS_instrument_ID(inst)
                    classificationReport.expTime = (header['EXPTIME'])
                    classificationReport.observers = 'LTRobot'
                    classificationReport.reducers = 'D. Perley'
                    classificationReport.specTypeID = spectype_id
                    classificationReport.spectrumComments = spec_comments
                    classificationReport.groupID = source_group
                    classificationReport.spec_proprietary_period_value = proprietary_period
                    classificationReport.spec_proprietary_period_units = proprietary_units

                    pprint(classificationReport.fill(), tab='  ')
                    proceed = input("\nProceed with classification and upload? ([y]/n) : ")
                    if proceed == 'y' and not proceed.strip() == '':

                        # ASCII FILE UPLOAD
                        print ("\n")
                        response = upload_to_TNS(files)
                        print (response)

                        if not response:
                            print("File upload didn't work")
                            print(response)
                            #return False

                        print(response['id_code'], response['id_message'],
                              "\nSuccessfully uploaded ascii spectrum")
                        #classificationReport.asciiName = response['data'][-1]

                        report_id = tns_classify(classificationReport)
                        post_comment(ztfname, 'Uploaded to TNS')
                        tns_feedback(report_id)



                if inst == 'ALFOSC':

                    classifiers = 'A. Dahiwale, C. Fremling(Caltech) on behalf of the Zwicky Transient Facility (ZTF)'### Change accordingly
                    source_group = 48 ### Require source group id from drop down list, 0 is for None
                    spectypes = np.array(['object','host','sky','arcs','synthetic'])

                    #proprietary_period = int(input("Proprietary period in years:", x)
                    proprietary_period = '0'
                    proprietary_units = "years"
                    spec_comments =''
                    classification_comments = ''
                    spectype='object'
                    spectype_id = ['object', 'host', 'sky', 'arcs', 'synthetic'].index(spectype) + 1

                    header = (a['data']['altdata'])
                    obsdate = str(a['data']['observed_at'].split('T')[0])+' '+str(a['data']['observed_at'].split('T')[1])

                    classificationReport = TNSClassificationReport()
                    classificationReport.name = get_IAUname(ztfname)[0]['name'][3:]
                    classificationReport.fitsName = ''
                    classificationReport.asciiName = spectrum_name
                    classificationReport.classifierName = classifiers
                    classificationReport.classificationID = get_TNS_classification_ID(classify)
                    classificationReport.redshift = get_redshift(ztfname)
                    classificationReport.classificationComments = classification_comments
                    classificationReport.obsDate = obsdate
                    classificationReport.instrumentID = get_TNS_instrument_ID(inst)
                    classificationReport.expTime = (header['EXPTIME'])
                    classificationReport.observers = (str(a['data']['observers'][0]['first_name'])+' '+str(a['data']['observers'][0]['last_name']))
                    classificationReport.reducers = (str(a['data']['reducers'][0]['first_name'])+' '+str(a['data']['reducers'][0]['last_name']))
                    classificationReport.specTypeID = spectype_id
                    classificationReport.spectrumComments = spec_comments
                    classificationReport.groupID = source_group
                    classificationReport.spec_proprietary_period_value = proprietary_period
                    classificationReport.spec_proprietary_period_units = proprietary_units

                    pprint(classificationReport.fill(), tab='  ')
                    proceed = input("\nProceed with classification and upload? ([y]/n) : ")
                    if proceed == 'y' and not proceed.strip() == '':

                        # ASCII FILE UPLOAD
                        print ("\n")
                        response = upload_to_TNS(files)
                        print (response)

                        if not response:
                            print("File upload didn't work")
                            print(response)
                            #return False

                        print(response['id_code'], response['id_message'],
                              "\nSuccessfully uploaded ascii spectrum")
                        #classificationReport.asciiName = response['data'][-1]

                        report_id = tns_classify(classificationReport)
                        post_comment(ztfname, 'Uploaded to TNS')
                        tns_feedback(report_id)



                if inst == 'DBSP':

                    classifiers = 'A. Dahiwale, C. Fremling(Caltech) on behalf of the Zwicky Transient Facility (ZTF)'### Change accordingly
                    source_group = 48 ### Require source group id from drop down list, 0 is for None
                    spectypes = np.array(['object','host','sky','arcs','synthetic'])

                    #proprietary_period = int(input("Proprietary period in years:", x)
                    proprietary_period = '0'
                    proprietary_units = "years"
                    spec_comments =''
                    classification_comments = ''
                    spectype='object'
                    spectype_id = ['object', 'host', 'sky', 'arcs', 'synthetic'].index(spectype) + 1

                    OBSDATE = str(a['data']['observed_at'].split('T')[0])+' '+str(a['data']['observed_at'].split('T')[1])

                    classificationReport = TNSClassificationReport()
                    classificationReport.name = get_IAUname(ztfname)[0]['name'][3:]
                    classificationReport.fitsName = ''
                    classificationReport.asciiName = spectrum_name
                    classificationReport.classifierName = classifiers
                    classificationReport.classificationID = get_TNS_classification_ID(classify)
                    classificationReport.redshift = get_redshift(ztfname)
                    classificationReport.classificationComments = classification_comments
                    classificationReport.obsDate = OBSDATE
                    classificationReport.instrumentID = get_TNS_instrument_ID(inst)
                    #classificationReport.expTime = '900'
                    classificationReport.observers = (str(a['data']['observers'][0]['first_name'])+' '+str(a['data']['observers'][0]['last_name']))
                    classificationReport.reducers = (str(a['data']['reducers'][0]['first_name'])+' '+str(a['data']['reducers'][0]['last_name']))
                    classificationReport.specTypeID = spectype_id
                    classificationReport.spectrumComments = spec_comments
                    classificationReport.groupID = source_group
                    classificationReport.spec_proprietary_period_value = proprietary_period
                    classificationReport.spec_proprietary_period_units = proprietary_units

                    pprint(classificationReport.fill(), tab='  ')
                    proceed = input("\nProceed with classification and upload? ([y]/n) : ")
                    if proceed == 'y' and not proceed.strip() == '':

                        # ASCII FILE UPLOAD
                        print ("\n")
                        response = upload_to_TNS(files)
                        print (response)

                        if not response:
                            print("File upload didn't work")
                            print(response)
                            #return False

                        print(response['id_code'], response['id_message'],
                              "\nSuccessfully uploaded ascii spectrum")
                        #classificationReport.asciiName = response['data'][-1]

                        report_id = tns_classify(classificationReport)
                        post_comment(ztfname, 'Uploaded to TNS')
                        tns_feedback(report_id)


                if inst == 'LRIS':

                    classifiers = 'A. Dahiwale, C. Fremling(Caltech) on behalf of the Zwicky Transient Facility (ZTF)'### Change accordingly
                    source_group = 48 ### Require source group id from drop down list, 0 is for None
                    spectypes = np.array(['object','host','sky','arcs','synthetic'])

                    header = (a['data']['altdata'])

                    #proprietary_period = int(input("Proprietary period in years:", x)
                    proprietary_period = '0'
                    proprietary_units = "years"
                    spec_comments =''
                    classification_comments = ''
                    spectype='object'
                    spectype_id = ['object', 'host', 'sky', 'arcs', 'synthetic'].index(spectype) + 1

                    OBSDATE = str(a['data']['observed_at'].split('T')[0])+' '+str(a['data']['observed_at'].split('T')[1])

                    classificationReport = TNSClassificationReport()
                    classificationReport.name = get_IAUname(ztfname)[0]['name'][3:]
                    classificationReport.fitsName = ''
                    classificationReport.asciiName = spectrum_name
                    classificationReport.classifierName = classifiers
                    classificationReport.classificationID = get_TNS_classification_ID(classify)
                    classificationReport.redshift = get_redshift(ztfname)
                    classificationReport.classificationComments = classification_comments
                    classificationReport.obsDate = OBSDATE
                    classificationReport.instrumentID = get_TNS_instrument_ID(inst)
                    classificationReport.expTime = '300'
                    classificationReport.observers = (str(a['data']['observers'][0]['first_name'])+' '+str(a['data']['observers'][0]['last_name']))
                    classificationReport.reducers = (str(a['data']['reducers'][0]['first_name'])+' '+str(a['data']['reducers'][0]['last_name']))
                    classificationReport.specTypeID = spectype_id
                    classificationReport.spectrumComments = spec_comments
                    classificationReport.groupID = source_group
                    classificationReport.spec_proprietary_period_value = proprietary_period
                    classificationReport.spec_proprietary_period_units = proprietary_units

                    pprint(classificationReport.fill(), tab='  ')
                    proceed = input("\nProceed with classification and upload? ([y]/n) : ")
                    if proceed == 'y' and not proceed.strip() == '':

                        # ASCII FILE UPLOAD
                        print ("\n")
                        response = upload_to_TNS(files)
                        print (response)

                        if not response:
                            print("File upload didn't work")
                            print(response)
                            #return False

                        print(response['id_code'], response['id_message'],
                              "\nSuccessfully uploaded ascii spectrum")
                        #classificationReport.asciiName = response['data'][-1]

                        report_id = tns_classify(classificationReport)
                        post_comment(ztfname, 'Uploaded to TNS')
                        tns_feedback(report_id)


                if inst == 'DIS':

                    classifiers = 'Melissa L. Graham (UW), A. Dahiwale, C. Fremling(Caltech) on behalf of the Zwicky Transient Facility (ZTF)'### Change accordingly
                    source_group = 48 ### Require source group id from drop down list, 0 is for None
                    spectypes = np.array(['object','host','sky','arcs','synthetic'])
                    #proprietary_period = int(input("Proprietary period in years:", x)
                    proprietary_period = '0'
                    proprietary_units = "years"
                    spec_comments =''
                    classification_comments = ''
                    spectype='object'
                    spectype_id = ['object', 'host', 'sky', 'arcs', 'synthetic'].index(spectype) + 1

                    #obsdate = APO(specid)[0]
                    #exptime = APO(specid)[1]
                    #observers = APO(specid)[2]
                    #reducers = APO(specid)[3]

                    obsdate = str(a['data']['observed_at'].split('T')[0])+' '+str(a['data']['observed_at'].split('T')[1])

                    classificationReport = TNSClassificationReport()
                    classificationReport.name = get_IAUname(ztfname)[0]['name'][3:]
                    classificationReport.fitsName = ''
                    classificationReport.asciiName = spectrum_name
                    classificationReport.classifierName = classifiers
                    classificationReport.classificationID = get_TNS_classification_ID(classify)
                    classificationReport.redshift = get_redshift(ztfname)
                    classificationReport.classificationComments = classification_comments
                    classificationReport.obsDate = obsdate
                    classificationReport.instrumentID = get_TNS_instrument_ID(inst)
                    #classificationReport.expTime = exptime
                    #classificationReport.observers = observers
                    #classificationReport.reducers = reducers
                    classificationReport.observers = (str(a['data']['observers'][0]['first_name'])+' '+str(a['data']['observers'][0]['last_name']))
                    classificationReport.reducers = (str(a['data']['reducers'][0]['first_name'])+' '+str(a['data']['reducers'][0]['last_name']))
                    classificationReport.specTypeID = spectype_id
                    classificationReport.spectrumComments = spec_comments
                    classificationReport.groupID = source_group
                    classificationReport.spec_proprietary_period_value = proprietary_period
                    classificationReport.spec_proprietary_period_units = proprietary_units

                    pprint(classificationReport.fill(), tab='  ')

                    proceed = input("\nProceed with classification and upload? ([y]/n) : ")
                    if proceed == 'y' and not proceed.strip() == '':

                        #ASCII FILE UPLOAD
                        print ("\n")
                        response = upload_to_TNS(files)
                        print (response)

                        if not response:
                            print("File upload didn't work")
                            print(response)
                            #return False

                        print(response['id_code'], response['id_message'],
                              "\nSuccessfully uploaded ascii spectrum")
                        #classificationReport.asciiName = response['data'][-1]

                        report_id = tns_classify(classificationReport)
                        post_comment(ztfname, 'Uploaded to TNS')
                        tns_feedback(report_id)
