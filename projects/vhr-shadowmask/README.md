Formulas tried

Try #1
x_cld = cX + H * tan(raster_obj.MEANSATEL) * sin(raster_obj.MEANSATAZ)
y_cld = cY + H * tan(raster_obj.MEANSATEL) * cos(raster_obj.MEANSATAZ)

x_shd = x_cld + H * tan(raster_obj.MEANSUNEL) * sin(raster_obj.MEANSUNAZ)
y_shd = y_cld + H * tan(raster_obj.MEANSUNEL) * cos(raster_obj.MEANSUNAZ)

Try #1
x_cld = cX + H * tan(raster_obj.MEANSATEL) * sin(raster_obj.MEANSATAZ)
y_cld = cY + H * tan(raster_obj.MEANSATEL) * cos(raster_obj.MEANSATAZ)

x_shd = x_cld + H * tan(raster_obj.MEANSUNEL) * sin(raster_obj.MEANSUNAZ)
y_shd = y_cld + H * tan(raster_obj.MEANSUNEL) * cos(raster_obj.MEANSUNAZ)