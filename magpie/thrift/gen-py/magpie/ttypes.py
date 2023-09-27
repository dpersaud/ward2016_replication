#
# Autogenerated by Thrift Compiler (0.9.2)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py
#

from thrift.Thrift import TType, TMessageType, TException, TApplicationException

from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol, TProtocol
try:
  from thrift.protocol import fastbinary
except:
  fastbinary = None



class Entry:
  """
  Store data regarding an entry:

   name                : String describing this entry (parsed before
   measuredProperties  : Measured class variable
   predictedProperties : Predicted class variable for each model
   classProbs          : (Classifiers) Probability of membership in each class

  Attributes:
   - name
   - measuredProperties
   - predictedProperties
   - classProbs
  """

  thrift_spec = (
    None, # 0
    (1, TType.STRING, 'name', None, None, ), # 1
    (2, TType.MAP, 'measuredProperties', (TType.STRING,None,TType.DOUBLE,None), {
    }, ), # 2
    (3, TType.MAP, 'predictedProperties', (TType.STRING,None,TType.DOUBLE,None), {
    }, ), # 3
    (4, TType.MAP, 'classProbs', (TType.STRING,None,TType.LIST,(TType.DOUBLE,None)), {
    }, ), # 4
  )

  def __init__(self, name=None, measuredProperties=thrift_spec[2][4], predictedProperties=thrift_spec[3][4], classProbs=thrift_spec[4][4],):
    self.name = name
    if measuredProperties is self.thrift_spec[2][4]:
      measuredProperties = {
    }
    self.measuredProperties = measuredProperties
    if predictedProperties is self.thrift_spec[3][4]:
      predictedProperties = {
    }
    self.predictedProperties = predictedProperties
    if classProbs is self.thrift_spec[4][4]:
      classProbs = {
    }
    self.classProbs = classProbs

  def read(self, iprot):
    if iprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None and fastbinary is not None:
      fastbinary.decode_binary(self, iprot.trans, (self.__class__, self.thrift_spec))
      return
    iprot.readStructBegin()
    while True:
      (fname, ftype, fid) = iprot.readFieldBegin()
      if ftype == TType.STOP:
        break
      if fid == 1:
        if ftype == TType.STRING:
          self.name = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 2:
        if ftype == TType.MAP:
          self.measuredProperties = {}
          (_ktype1, _vtype2, _size0 ) = iprot.readMapBegin()
          for _i4 in xrange(_size0):
            _key5 = iprot.readString();
            _val6 = iprot.readDouble();
            self.measuredProperties[_key5] = _val6
          iprot.readMapEnd()
        else:
          iprot.skip(ftype)
      elif fid == 3:
        if ftype == TType.MAP:
          self.predictedProperties = {}
          (_ktype8, _vtype9, _size7 ) = iprot.readMapBegin()
          for _i11 in xrange(_size7):
            _key12 = iprot.readString();
            _val13 = iprot.readDouble();
            self.predictedProperties[_key12] = _val13
          iprot.readMapEnd()
        else:
          iprot.skip(ftype)
      elif fid == 4:
        if ftype == TType.MAP:
          self.classProbs = {}
          (_ktype15, _vtype16, _size14 ) = iprot.readMapBegin()
          for _i18 in xrange(_size14):
            _key19 = iprot.readString();
            _val20 = []
            (_etype24, _size21) = iprot.readListBegin()
            for _i25 in xrange(_size21):
              _elem26 = iprot.readDouble();
              _val20.append(_elem26)
            iprot.readListEnd()
            self.classProbs[_key19] = _val20
          iprot.readMapEnd()
        else:
          iprot.skip(ftype)
      else:
        iprot.skip(ftype)
      iprot.readFieldEnd()
    iprot.readStructEnd()

  def write(self, oprot):
    if oprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and self.thrift_spec is not None and fastbinary is not None:
      oprot.trans.write(fastbinary.encode_binary(self, (self.__class__, self.thrift_spec)))
      return
    oprot.writeStructBegin('Entry')
    if self.name is not None:
      oprot.writeFieldBegin('name', TType.STRING, 1)
      oprot.writeString(self.name)
      oprot.writeFieldEnd()
    if self.measuredProperties is not None:
      oprot.writeFieldBegin('measuredProperties', TType.MAP, 2)
      oprot.writeMapBegin(TType.STRING, TType.DOUBLE, len(self.measuredProperties))
      for kiter27,viter28 in self.measuredProperties.items():
        oprot.writeString(kiter27)
        oprot.writeDouble(viter28)
      oprot.writeMapEnd()
      oprot.writeFieldEnd()
    if self.predictedProperties is not None:
      oprot.writeFieldBegin('predictedProperties', TType.MAP, 3)
      oprot.writeMapBegin(TType.STRING, TType.DOUBLE, len(self.predictedProperties))
      for kiter29,viter30 in self.predictedProperties.items():
        oprot.writeString(kiter29)
        oprot.writeDouble(viter30)
      oprot.writeMapEnd()
      oprot.writeFieldEnd()
    if self.classProbs is not None:
      oprot.writeFieldBegin('classProbs', TType.MAP, 4)
      oprot.writeMapBegin(TType.STRING, TType.LIST, len(self.classProbs))
      for kiter31,viter32 in self.classProbs.items():
        oprot.writeString(kiter31)
        oprot.writeListBegin(TType.DOUBLE, len(viter32))
        for iter33 in viter32:
          oprot.writeDouble(iter33)
        oprot.writeListEnd()
      oprot.writeMapEnd()
      oprot.writeFieldEnd()
    oprot.writeFieldStop()
    oprot.writeStructEnd()

  def validate(self):
    return


  def __hash__(self):
    value = 17
    value = (value * 31) ^ hash(self.name)
    value = (value * 31) ^ hash(self.measuredProperties)
    value = (value * 31) ^ hash(self.predictedProperties)
    value = (value * 31) ^ hash(self.classProbs)
    return value

  def __repr__(self):
    L = ['%s=%r' % (key, value)
      for key, value in self.__dict__.iteritems()]
    return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

  def __eq__(self, other):
    return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

  def __ne__(self, other):
    return not (self == other)

class ModelInfo:
  """
  Holds all known information about a model

  Known properties:
   author      : Name/contact info of author
   citation    : Citation information of the model
   classifier  : Whether this model is a classification (or regression) model
   dataType    : Type of data expected, defined by name of Magpie Dataset type
   description : Short description of this model
   modelType   : Simple description of model
   notes       : Any pertinent details about the model
   property    : Property being modeled
   training    : Description of training set
   trainTime   : When this model was trained (formatted string)
   units       : Units of prediction. (Classifiers) Name of classes, ";"-delimited
   valMethod   : Description of how this model was validated
   valScore    : Performance of model in cross-validation tests

  Attributes:
   - property
   - units
   - author
   - training
   - citation
   - notes
   - dataType
   - modelType
   - classifier
   - valScore
   - description
   - valMethod
   - trainTime
  """

  thrift_spec = (
    None, # 0
    (1, TType.STRING, 'property', None, None, ), # 1
    (2, TType.STRING, 'units', None, None, ), # 2
    (3, TType.STRING, 'author', None, None, ), # 3
    (4, TType.STRING, 'training', None, None, ), # 4
    (5, TType.STRING, 'citation', None, None, ), # 5
    (6, TType.STRING, 'notes', None, None, ), # 6
    (7, TType.STRING, 'dataType', None, None, ), # 7
    (8, TType.STRING, 'modelType', None, None, ), # 8
    (9, TType.BOOL, 'classifier', None, None, ), # 9
    (10, TType.MAP, 'valScore', (TType.STRING,None,TType.DOUBLE,None), None, ), # 10
    (11, TType.STRING, 'description', None, None, ), # 11
    (12, TType.STRING, 'valMethod', None, None, ), # 12
    (13, TType.STRING, 'trainTime', None, None, ), # 13
  )

  def __init__(self, property=None, units=None, author=None, training=None, citation=None, notes=None, dataType=None, modelType=None, classifier=None, valScore=None, description=None, valMethod=None, trainTime=None,):
    self.property = property
    self.units = units
    self.author = author
    self.training = training
    self.citation = citation
    self.notes = notes
    self.dataType = dataType
    self.modelType = modelType
    self.classifier = classifier
    self.valScore = valScore
    self.description = description
    self.valMethod = valMethod
    self.trainTime = trainTime

  def read(self, iprot):
    if iprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None and fastbinary is not None:
      fastbinary.decode_binary(self, iprot.trans, (self.__class__, self.thrift_spec))
      return
    iprot.readStructBegin()
    while True:
      (fname, ftype, fid) = iprot.readFieldBegin()
      if ftype == TType.STOP:
        break
      if fid == 1:
        if ftype == TType.STRING:
          self.property = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 2:
        if ftype == TType.STRING:
          self.units = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 3:
        if ftype == TType.STRING:
          self.author = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 4:
        if ftype == TType.STRING:
          self.training = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 5:
        if ftype == TType.STRING:
          self.citation = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 6:
        if ftype == TType.STRING:
          self.notes = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 7:
        if ftype == TType.STRING:
          self.dataType = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 8:
        if ftype == TType.STRING:
          self.modelType = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 9:
        if ftype == TType.BOOL:
          self.classifier = iprot.readBool();
        else:
          iprot.skip(ftype)
      elif fid == 10:
        if ftype == TType.MAP:
          self.valScore = {}
          (_ktype35, _vtype36, _size34 ) = iprot.readMapBegin()
          for _i38 in xrange(_size34):
            _key39 = iprot.readString();
            _val40 = iprot.readDouble();
            self.valScore[_key39] = _val40
          iprot.readMapEnd()
        else:
          iprot.skip(ftype)
      elif fid == 11:
        if ftype == TType.STRING:
          self.description = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 12:
        if ftype == TType.STRING:
          self.valMethod = iprot.readString();
        else:
          iprot.skip(ftype)
      elif fid == 13:
        if ftype == TType.STRING:
          self.trainTime = iprot.readString();
        else:
          iprot.skip(ftype)
      else:
        iprot.skip(ftype)
      iprot.readFieldEnd()
    iprot.readStructEnd()

  def write(self, oprot):
    if oprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and self.thrift_spec is not None and fastbinary is not None:
      oprot.trans.write(fastbinary.encode_binary(self, (self.__class__, self.thrift_spec)))
      return
    oprot.writeStructBegin('ModelInfo')
    if self.property is not None:
      oprot.writeFieldBegin('property', TType.STRING, 1)
      oprot.writeString(self.property)
      oprot.writeFieldEnd()
    if self.units is not None:
      oprot.writeFieldBegin('units', TType.STRING, 2)
      oprot.writeString(self.units)
      oprot.writeFieldEnd()
    if self.author is not None:
      oprot.writeFieldBegin('author', TType.STRING, 3)
      oprot.writeString(self.author)
      oprot.writeFieldEnd()
    if self.training is not None:
      oprot.writeFieldBegin('training', TType.STRING, 4)
      oprot.writeString(self.training)
      oprot.writeFieldEnd()
    if self.citation is not None:
      oprot.writeFieldBegin('citation', TType.STRING, 5)
      oprot.writeString(self.citation)
      oprot.writeFieldEnd()
    if self.notes is not None:
      oprot.writeFieldBegin('notes', TType.STRING, 6)
      oprot.writeString(self.notes)
      oprot.writeFieldEnd()
    if self.dataType is not None:
      oprot.writeFieldBegin('dataType', TType.STRING, 7)
      oprot.writeString(self.dataType)
      oprot.writeFieldEnd()
    if self.modelType is not None:
      oprot.writeFieldBegin('modelType', TType.STRING, 8)
      oprot.writeString(self.modelType)
      oprot.writeFieldEnd()
    if self.classifier is not None:
      oprot.writeFieldBegin('classifier', TType.BOOL, 9)
      oprot.writeBool(self.classifier)
      oprot.writeFieldEnd()
    if self.valScore is not None:
      oprot.writeFieldBegin('valScore', TType.MAP, 10)
      oprot.writeMapBegin(TType.STRING, TType.DOUBLE, len(self.valScore))
      for kiter41,viter42 in self.valScore.items():
        oprot.writeString(kiter41)
        oprot.writeDouble(viter42)
      oprot.writeMapEnd()
      oprot.writeFieldEnd()
    if self.description is not None:
      oprot.writeFieldBegin('description', TType.STRING, 11)
      oprot.writeString(self.description)
      oprot.writeFieldEnd()
    if self.valMethod is not None:
      oprot.writeFieldBegin('valMethod', TType.STRING, 12)
      oprot.writeString(self.valMethod)
      oprot.writeFieldEnd()
    if self.trainTime is not None:
      oprot.writeFieldBegin('trainTime', TType.STRING, 13)
      oprot.writeString(self.trainTime)
      oprot.writeFieldEnd()
    oprot.writeFieldStop()
    oprot.writeStructEnd()

  def validate(self):
    return


  def __hash__(self):
    value = 17
    value = (value * 31) ^ hash(self.property)
    value = (value * 31) ^ hash(self.units)
    value = (value * 31) ^ hash(self.author)
    value = (value * 31) ^ hash(self.training)
    value = (value * 31) ^ hash(self.citation)
    value = (value * 31) ^ hash(self.notes)
    value = (value * 31) ^ hash(self.dataType)
    value = (value * 31) ^ hash(self.modelType)
    value = (value * 31) ^ hash(self.classifier)
    value = (value * 31) ^ hash(self.valScore)
    value = (value * 31) ^ hash(self.description)
    value = (value * 31) ^ hash(self.valMethod)
    value = (value * 31) ^ hash(self.trainTime)
    return value

  def __repr__(self):
    L = ['%s=%r' % (key, value)
      for key, value in self.__dict__.iteritems()]
    return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

  def __eq__(self, other):
    return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

  def __ne__(self, other):
    return not (self == other)

class MagpieException(TException):
  """
  Attributes:
   - why
  """

  thrift_spec = (
    None, # 0
    (1, TType.STRING, 'why', None, None, ), # 1
  )

  def __init__(self, why=None,):
    self.why = why

  def read(self, iprot):
    if iprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None and fastbinary is not None:
      fastbinary.decode_binary(self, iprot.trans, (self.__class__, self.thrift_spec))
      return
    iprot.readStructBegin()
    while True:
      (fname, ftype, fid) = iprot.readFieldBegin()
      if ftype == TType.STOP:
        break
      if fid == 1:
        if ftype == TType.STRING:
          self.why = iprot.readString();
        else:
          iprot.skip(ftype)
      else:
        iprot.skip(ftype)
      iprot.readFieldEnd()
    iprot.readStructEnd()

  def write(self, oprot):
    if oprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and self.thrift_spec is not None and fastbinary is not None:
      oprot.trans.write(fastbinary.encode_binary(self, (self.__class__, self.thrift_spec)))
      return
    oprot.writeStructBegin('MagpieException')
    if self.why is not None:
      oprot.writeFieldBegin('why', TType.STRING, 1)
      oprot.writeString(self.why)
      oprot.writeFieldEnd()
    oprot.writeFieldStop()
    oprot.writeStructEnd()

  def validate(self):
    return


  def __str__(self):
    return repr(self)

  def __hash__(self):
    value = 17
    value = (value * 31) ^ hash(self.why)
    return value

  def __repr__(self):
    L = ['%s=%r' % (key, value)
      for key, value in self.__dict__.iteritems()]
    return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

  def __eq__(self, other):
    return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

  def __ne__(self, other):
    return not (self == other)
