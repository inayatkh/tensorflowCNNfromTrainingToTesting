'''
Created on Sep 21, 2017

@author: inayat
'''


try:
    import json
except ImportError:
    # If python version is 2.5 or less, use simplejson
    import simplejson as json

import re
import traceback



def loads(text, **kwargs):
    ''' Deserialize `text` (a `str` or `unicode` instance containing a JSON
    document with Python or JavaScript like comments) to a Python object.

    :param text: serialized JSON string with or without comments.
    :param kwargs: all the arguments that `json.loads <http://docs.python.org/
                   2/library/json.html#json.loads>`_ accepts.
    :raises: commentjson.JSONLibraryException
    :returns: dict or list.
    '''
    regex = r'\s*(#|\/{2}).*$'
    regex_inline = r'(:?(?:\s)*([A-Za-z\d\.{}]*)|((?<=\").*\"),?)(?:\s)*(((#|(\/{2})).*)|)$'
    lines = text.split('\n')

    for index, line in enumerate(lines):
        if re.search(regex, line):
            if re.search(r'^' + regex, line, re.IGNORECASE):
                lines[index] = ""
            elif re.search(regex_inline, line):
                lines[index] = re.sub(regex_inline, r'\1', line)

    try:
        return json.loads('\n'.join(lines), **kwargs)
    except (Exception, e):
        raise JSONLibraryException(e.message)

class Confcommentjson(object):
    '''
    classdocs
    '''


    def __init__(self, jsonConfFilePath):
        '''
         load and store the .json configuration file and update object dictionary
        '''
        conf = loads(open(jsonConfFilePath).read())
        
        
        self.__dict__.update(conf)
        
    def __getitem__(self, key):
        '''
         return the value associated with the supplied , from the json conf file
         
        '''
        
        return self.__dict__.get(key, None)
        
        
class JSONLibraryException(Exception):
    ''' Exception raised when the JSON library in use raises an exception i.e.
    the exception is not caused by `commentjson` and only caused by the JSON
    library `commentjson` is using.

    .. note::

        As of now, ``commentjson`` supports only standard library's ``json``
        module. It might start supporting other widely-used contributed JSON
        libraries in the future.
    '''

    def __init__(self, json_error=""):
        tb = traceback.format_exc()
        tb = '\n'.join(' ' * 4 + line_ for line_ in tb.split('\n'))
        message = [
            'JSON Library Exception\n',
            ('Exception thrown by JSON library (json): '
             '\033[4;37m%s\033[0m\n' % json_error),
            '%s' % tb,
        ]
        Exception.__init__(self, '\n'.join(message))
