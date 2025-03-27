### Kornpob Bhirombhakdi
### kbhirombhakdi@stsci.edu

from astropy import wcs
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from polynomial2d.polynomial2d import Polynomial2D
import os
from photutils import CircularAperture
from photutils import aperture_photometry
from photutils.utils import calc_total_error
from hstapcorr.photapcorr import PhotApCorr

class Photometry:
    '''
    Photometry is a class containing methods for computing photometry with local 2D-polynomial background estimation.
    obj = Photometry() to instantiate
    obj.filename to see filename
    obj.root to see root name
    obj.skycoord to see skycoord of the object
    obj.wcs to see wcs
    obj.xy to see (pixX,pixY) of the object converting from skycoord and wcs
        call Photometry(pixXY = (pixX,pixY)) to use the specified pixXY instead of reading from WCS
    obj.pad to see padding dimensions
    obj.compute_bbox() to compute bbox
    obj.bbox to see different bounding boxes computed from pad and xy
    obj.prep_polynomial2d() prepares 2D polynomial object for background fitting
    obj.fit_polynomial2d() to estimate the background with 2D polynomial
    obj.polynomial2d to see 2D-polynomial input and background estimation
    obj.savesub() to save background subtracted image
    obj.subfile to see filepath of the background subtracted image
    obj.photometry() to compute photometry
    obj.phot_table to see photometry info
    obj.show2d() to see image plots
    obj.show3d() to see 3d plots
    obj.showsub() to see plots related to background subtraction
    '''
    def __init__(self,filename=None,skycoord=None,wcs=None,pixXY=None,
                 aprad=5,daprad=1,
                 padxleft=3,padxright=3,
                 padylow=3,padyup=3
                ):
        self.filename = filename
        try:
            self.root = filename.split('/')[-1].split('_')[0]
        except:
            self.root = None
        self.skycoord = skycoord
        self.wcs = wcs
        if pixXY:
            self.xy = pixXY
        else:
            self.xy = self._xy()
        self.pad = {'aprad':aprad,'daprad':daprad,
                    'padxleft':padxleft,'padxright':padxright,
                    'padylow':padylow,'padyup':padyup
                   }
        try:
            self.bbox = self.compute_bbox()
        except:
            self.bbox = None
        self.subfile = None
        self.phot_table = None
        self.polynomial2d = None
    def _xy(self):
        try:
            pixx,pixy = wcs.utils.skycoord_to_pixel(self.skycoord,self.wcs)
            return (pixx,pixy)
        except:
            return None
    ##########
    ##########
    ##########
    def photometry(self,instrument,save=True):
        pixx,pixy = self.xy
        aprad = self.pad['aprad']
        aperture = CircularAperture((pixx,pixy),r=aprad)
        tmp = fits.open(self.subfile)
        tmpdata = tmp[1].data
        exptime = tmp[0].header['EXPTIME']
        dateobs = tmp[0].header['DATE-OBS']
        filt = tmp[0].header['FILTER']
        error = calc_total_error(tmpdata,
                                 bkg_error=np.zeros_like(tmpdata),
                                 effective_gain=exptime
                                )
        phot_table = aperture_photometry(tmpdata,aperture,error=error)
        photapcorr = PhotApCorr()
        wave,zp = photapcorr.table[instrument]['ZP'][filt]
        scale = photapcorr.table[instrument]['scale']
        apsize = aprad * scale
        apcorr = photapcorr.table[instrument]['model'](wave,apsize)[0]
        mag = -2.5 * np.log10(phot_table['aperture_sum'] / apcorr) + zp
        emag = 2.5 * np.sqrt(phot_table['aperture_sum_err']**2) / (phot_table['aperture_sum'] * np.log(10.))        
        tmp = {'phot_table':phot_table,
               'apcorr':apcorr,
               'mag':(mag[0],emag[0]),
               'wavelength':wave,
               'zp':zp
              }
        self.phot_table = tmp
        ##########
        ##########
        ##########
        plt.figure()
        plt.imshow(tmpdata,origin='lower',cmap='viridis',vmin=-1,vmax=1)
        aperture.plot(color='red')
        dx,dy = 3,3
        plt.xlim(pixx-aprad-dx,pixx+aprad+dx)
        plt.ylim(pixy-aprad-dy,pixy+aprad+dy)
        string = '{0}\n{1}\n{2}\n{3:.3f} +/- {4:.3f} mag'.format(self.root,dateobs,filt,mag[0],emag[0])
        plt.title(string,fontsize=14)
        plt.tight_layout()
        if save:
            string = '{0}_photometry.eps'.format(self.root)
            plt.savefig(string,format='eps', bbox_inches='tight')
            print('Save to {0}'.format(string))
    ##########
    ##########
    ##########    
    def savesub(self):
        os.system('cp {0} {1}_sub.fits'.format(self.filename,self.root))
        string = '{0}_sub.fits'.format(self.root)
        tmp = fits.open(string)
        tmpdata = tmp[1].data.copy()
        skyboxx = self.bbox['skyboxx']
        skyboxy = self.bbox['skyboxy']
        obj = self.polynomial2d
        tmpdata[skyboxy[0]:skyboxy[1],skyboxx[0]:skyboxx[1]] = obj.data['Y'] - obj.model['YFIT']
        tmp[1].data = tmpdata.copy()
        tmp.writeto(string,overwrite=True)
        print('Save to {0}'.format(string))
        self.subfile = string
    ##########
    ##########
    ##########        
    def fit_polynomial2d(self,norder=2):
        self.polynomial2d.model['NORDER'] = norder
        self.polynomial2d.fit()
        self.polynomial2d.compute()
    ##########
    ##########
    ##########
    def prep_polynomial2d(self):
        obj = Polynomial2D()
        tmp = fits.open(self.filename)
        skyboxx = self.bbox['skyboxx']
        skyboxy = self.bbox['skyboxy']
        padxleft = self.pad['padxleft']
        daprad = self.pad['daprad']
        aprad = self.pad['aprad']
        padylow = self.pad['padylow']
        tmpdata = tmp[1].data[skyboxy[0]:skyboxy[1],skyboxx[0]:skyboxx[1]]
        shapey,shapex = tmpdata.shape
        x1 = np.arange(shapex)
        x2 = np.arange(shapey)
        x1,x2 = np.meshgrid(x1,x2)
        obj.data['X1'] = x1.copy()
        obj.data['X2'] = x2.copy()
        obj.data['Y'] = tmpdata.copy()
        obj.data['MASK'] = np.full_like(tmpdata,False,dtype=bool)
        tmp = obj.data['MASK'].copy()
        tmpboxx = (padxleft,padxleft+daprad+aprad+1+aprad+daprad)
        tmpboxy = (padylow,padylow+daprad+aprad+1+aprad+daprad)
        tmp[tmpboxy[0]:tmpboxy[1],tmpboxx[0]:tmpboxx[1]] = True
        obj.model['MASKFIT'] = tmp.copy()
        self.polynomial2d = obj
    ##########
    ##########
    ##########
    def compute_bbox(self):
        aprad = self.pad['aprad']
        daprad = self.pad['daprad']
        padxleft,padxright = self.pad['padxleft'],self.pad['padxright']
        padylow,padyup = self.pad['padylow'],self.pad['padyup']
        pixx,pixy = self.xy
        inboxx = (int(np.round(pixx))-aprad,int(np.round(pixx))+aprad+1)
        inboxy = (int(np.round(pixy))-aprad,int(np.round(pixy))+aprad+1)
        outboxx = (inboxx[0]-daprad,inboxx[1]+daprad)
        outboxy = (inboxy[0]-daprad,inboxy[1]+daprad)
        skyboxx = (outboxx[0]-padxleft,outboxx[1]+padxright)
        skyboxy = (outboxy[0]-padylow,outboxy[1]+padyup)
        return {'inboxx':inboxx,'inboxy':inboxy,
                'outboxx':outboxx,'outboxy':outboxy,
                'skyboxx':skyboxx,'skyboxy':skyboxy
               }
    ##########
    ##########
    ##########
    def showsub(self,save=True):
        fig = plt.figure()
        obj = self.polynomial2d
        ax = fig.add_subplot(1,3,1,projection='3d')
        ax.plot_surface(obj.data['X1'],obj.data['X2'],obj.data['Y'],
                        cmap='rainbow',
                        alpha=0.6
                       )
        ax.plot_surface(obj.data['X1'],obj.data['X2'],obj.model['YFIT'],
                        cmap='gray',
                        alpha=1.
                       )
        ax.set_title('c/s')
        ax = fig.add_subplot(1,3,2,projection='3d')
        tmp = (obj.data['Y'] - obj.model['YFIT'])
        ax.plot_surface(obj.data['X1'],obj.data['X2'],tmp * ~obj.model['MASKFIT'],
                        cmap='rainbow',
                        alpha=0.6
                       )
        ax.plot_surface(obj.data['X1'],obj.data['X2'],np.full_like(tmp,0.,dtype=float),
                        cmap='gray',
                        alpha=1.
                       )
        m = np.where(obj.model['MASKFIT']==False)
        tmpmed = np.median(tmp[m])
        ax.set_title('SUB\nmedian = {0:.4f}'.format(tmpmed))
        ax = fig.add_subplot(1,3,3)
        tmp = obj.data['Y'] - obj.model['YFIT']
        vmin,vmax = np.percentile(tmp,5.),np.percentile(tmp,95.)
        ax.imshow(tmp,origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
        ax.set_title('SUB')
        plt.tight_layout()
        if save:
            string = '{0}_showsub.eps'.format(self.root)
            plt.savefig(string, format='eps', bbox_inches='tight')
            print('Save to {0}'.format(string))
    ##########
    ##########
    ##########
    def show3d(self,save=True):
        fig = plt.figure()
        ax = fig.add_subplot(1,4,1)
        obj = self.polynomial2d
        tmp = obj.data['Y']
        vmin,vmax = np.percentile(tmp,5.),np.percentile(tmp,95.)
        ax.imshow(tmp,origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
        ax = fig.add_subplot(1,4,2,projection='3d')
        ax.plot_surface(obj.data['X1'],obj.data['X2'],obj.data['Y'],
                        cmap='rainbow',
                        alpha=0.6
                       )
        ax.set_title('c/s')
        ax = fig.add_subplot(1,4,3,projection='3d')
        ax.plot_surface(obj.data['X1'],obj.data['X2'],obj.model['MASKFIT'],
                        cmap='rainbow',
                        alpha=0.6
                       )
        ax.set_title('MASKFIT')
        ax = fig.add_subplot(1,4,4,projection='3d')
        tmp = obj.data['Y'] * ~obj.model['MASKFIT']
        ax.plot_surface(obj.data['X1'],obj.data['X2'],tmp,
                        cmap='rainbow',
                        alpha=0.6
                       )
        ax.set_title('MASKED c/s')
        plt.tight_layout()
        if save:
            string = '{0}_show3d.eps'.format(self.root)
            plt.savefig(string, format='eps', bbox_inches='tight')
            print('Save to {0}'.format(string))
    ##########
    ##########
    ##########
    def show2d(self,save=True,bbox=False):
        fig = plt.figure()
        tmp = fits.open(self.filename)
        tmpdata = tmp[1].data
        tmpheader = tmp[0].header
        m = np.isfinite(tmpdata)
        vmin,vmax = np.percentile(tmpdata[m],5.),np.percentile(tmpdata[m],99.5)
        ax = fig.add_subplot(1,2,1)
        ax.imshow(tmpdata,origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
        ax.scatter(*self.xy,s=30,edgecolor='red',facecolor='None')
        string = '{0}\n{1}\nEXPSTART={2:.2f}\nEXPTIME={3:.2f}\n{4}'.format(self.root,
                                                                           tmpheader['DATE-OBS'],
                                                                           tmpheader['EXPSTART'],
                                                                           tmpheader['EXPTIME'],
                                                                           tmpheader['FILTER']
                                                                          )
        ax.set_title('{0}'.format(string),fontsize=14)
        ##########
        ##########
        ##########
        ax = fig.add_subplot(1,2,2)
        ax.imshow(tmpdata,origin='lower',cmap='viridis',vmin=vmin,vmax=vmax)
        ax.scatter(*self.xy,s=30,edgecolor='red',facecolor='None')
        dx,dy = 50,50
        pixx,pixy = self.xy
        ax.set_xlim(pixx-dx,pixx+dx)
        ax.set_ylim(pixy-dy,pixy+dy)
        RADEC = self.skycoord
        string = 'RA={0:.2f},DEC={1:.2f},\nx={2:.2f},y={3:.2f}'.format(RADEC.ra.deg,RADEC.dec.deg,pixx,pixy)
        ax.set_title('{0}'.format(string),fontsize=14)
        if bbox:
            tmpboxx,tmpboxy = self.bbox['inboxx'],self.bbox['inboxy']
            tmpx = (tmpboxx[0],tmpboxx[1],tmpboxx[1],tmpboxx[0],tmpboxx[0])
            tmpy = (tmpboxy[0],tmpboxy[0],tmpboxy[1],tmpboxy[1],tmpboxy[0])
            ax.plot(tmpx,tmpy,'r-')
            tmpboxx,tmpboxy = self.bbox['outboxx'],self.bbox['outboxy']
            tmpx = (tmpboxx[0],tmpboxx[1],tmpboxx[1],tmpboxx[0],tmpboxx[0])
            tmpy = (tmpboxy[0],tmpboxy[0],tmpboxy[1],tmpboxy[1],tmpboxy[0])
            ax.plot(tmpx,tmpy,'r-')
            tmpboxx,tmpboxy = self.bbox['skyboxx'],self.bbox['skyboxy']
            tmpx = (tmpboxx[0],tmpboxx[1],tmpboxx[1],tmpboxx[0],tmpboxx[0])
            tmpy = (tmpboxy[0],tmpboxy[0],tmpboxy[1],tmpboxy[1],tmpboxy[0])
            ax.plot(tmpx,tmpy,'r-')
        ##########
        ##########
        ##########
        if save:
            string = '{0}_show.eps'.format(self.root)
            plt.savefig(string, format='eps', bbox_inches='tight')  
            print('Save to {0}'.format(string))
        