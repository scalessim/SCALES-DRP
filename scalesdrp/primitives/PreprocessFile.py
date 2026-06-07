from keckdrpframework.primitives.base_primitive import BasePrimitive
from scalesdrp.primitives.scales_file_primitives import scales_fits_writer

from astropy.io import fits

class PreprocessFile(BasePrimitive):
    """
    Preprocess file for testing. Adds RA, DEC, IMTYPE, IFSMODE, PIXSCALE
    as necessary using default values if they cannot be derived from
    other keywords in the header.
    """

    def __init__(self, action, context):
        BasePrimitive.__init__(self, action, context)
        self.logger = context.pipeline_logger

    def _perform(self):

        fitsname = self.action.args.name
        outdir = self.config.instrument.output_directory

        with fits.open(fitsname, mode='update') as fitsfile:
            hdr = fitsfile[0]

            # try to retrieve each of the necessary keywords
            # and replace with a dummy value if they are not
            # included in the fitsfile. This is for testing
            # purposes only and should never replace real
            # values in the header file.
            ra = hdr.get('RA', 0)
            hdr['RA']=ra
            dec = hdr.get('DEC', '+00:00:00')
            hdr['DEC']=dec
            pixscale = hdr.get('PIXSCALE', 0.00107422)
            hdr['PIXSCALE']=pixscale
            imtype = self.deduce_imtype(hdr)
            hdr['IMTYPE']=imtype
            ifsmode = self.deduce_ifsmode(hdr)
            hdr['IFSMODE']=ifsmode

            scales_fits_writer(ccddata = opt_rslt, 
                table=self.action.args.table,
                output_file=self.action.args.name,
                output_dir=self.config.instrument.output_directory,
                suffix="opt_cube")
        return self.action.args

    def deduce_ifsmode(self, hdr):
        camera = hdr['CAMERA']
        ifsmode = ''
        if camera == 'Im':
            ifsmode = 'N/A'
        else:
            disperser = hdr['DSPRSNAM'].strip().split('-')[0]
            lenslet = hdr['LENSLNAM'].strip()
            ifsmode = '-'.join((lenslet, disperser))
        return ifsmode

    def deduce_imtype(self, hdr):
        imtype = 'OBJECT'
        return imtype

