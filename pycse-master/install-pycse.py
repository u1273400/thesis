<<<<<<< HEAD
# for installing magic IPython stuff
# In ipython run this
# %load https://raw.github.com/jkitchin/pycse/master/install-pycse.py

from setuptools.command import easy_install
easy_install.main( ["-U","pycse"] )

# my customized pyreport
package = 'https://github.com/jkitchin/pyreport/archive/master.zip'
easy_install.main( ["-U",package] )

import IPython, os
IPydir = os.path.join(IPython.utils.path.get_ipython_dir(),
                      'profile_default',
                      'startup')
                      
print 'Installing ipython magic to : ',IPydir

if not os.path.exists(IPydir):
    raise Exception('No ipython directory found')

url = 'https://raw.github.com/jkitchin/pycse/master/pycse/00-pycse-magic.py'

import urllib
urllib.urlretrieve (url, os.path.join(IPydir,'00-pycse-magic.py')) 
print 'Ipython magic installed now!'

# extra packages
easy_install.main( ["-U","quantities"] )
easy_install.main( ["-U","uncertainties"] )
print 'Extra packages now installed.'
=======
# for installing magic IPython stuff
# In ipython run this
# %load https://raw.github.com/jkitchin/pycse/master/install-pycse.py

from setuptools.command import easy_install
easy_install.main( ["-U","pycse"] )

# my customized pyreport
package = 'https://github.com/jkitchin/pyreport/archive/master.zip'
easy_install.main( ["-U",package] )

import IPython, os
IPydir = os.path.join(IPython.utils.path.get_ipython_dir(),
                      'profile_default',
                      'startup')
                      
print 'Installing ipython magic to : ',IPydir

if not os.path.exists(IPydir):
    raise Exception('No ipython directory found')

url = 'https://raw.github.com/jkitchin/pycse/master/pycse/00-pycse-magic.py'

import urllib
urllib.urlretrieve (url, os.path.join(IPydir,'00-pycse-magic.py')) 
print 'Ipython magic installed now!'

# extra packages
easy_install.main( ["-U","quantities"] )
easy_install.main( ["-U","uncertainties"] )
print 'Extra packages now installed.'
>>>>>>> 3ef68fb5b75bb45286846e57bc17926fa30d5ad1
