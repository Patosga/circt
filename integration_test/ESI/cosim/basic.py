#!/usr/bin/python3

import random
import cosim


class BasicSystemTester(cosim.CosimBase):
  """Provides methods to test the 'basic' simulation."""

  def testIntAcc(self, num_msgs):
    ep = self.openEP(1, sendType=self.schema.I32, recvType=self.schema.I32)
    sum = 0
    for _ in range(num_msgs):
      i = random.randint(0, 77)
      sum += i
      print(f"Sending {i}")
      ep.send(self.schema.I32.new_message(i=i))
      result = self.readMsg(ep, self.schema.I32)
      print(f"Got {result}")
      assert (result.i == sum)

  def testVectorSum(self, num_msgs):
    ep = self.openEP(2,
                     sendType=self.schema.ArrayOf2xUi24,
                     recvType=self.schema.ArrayOf4xSi13)
    for _ in range(num_msgs):
      # Since the result is unsigned, we need to make sure the sum is
      # never negative.
      arr = [
          random.randint(-468, 777),
          random.randint(500, 1250),
          random.randint(-468, 777),
          random.randint(500, 1250)
      ]
      print(f"Sending {arr}")
      ep.send(self.schema.ArrayOf4xSi13.new_message(l=arr))
      result = self.readMsg(ep, self.schema.ArrayOf2xUi24)
      print(f"Got {result}")
      assert (result.l[0] == arr[0] + arr[1])
      assert (result.l[1] == arr[2] + arr[3])

  def testCrypto(self, num_msgs):
    ep = self.openEP(3,
                     sendType=self.schema.Struct12811887160382076992,
                     recvType=self.schema.Struct12811887160382076992)
    cfg = self.openEP(4,
                      sendType=self.schema.I1,
                      recvType=self.schema.Struct14590566522150786282)

    cfgWritten = False
    for _ in range(num_msgs):
      blob = [random.randint(0, 255) for x in range(32)]
      print(f"Sending data {blob}")
      ep.send(
          self.schema.Struct12811887160382076992.new_message(encrypted=False,
                                                             blob=blob))

      if not cfgWritten:
        # Check that messages queue up properly waiting for the config.
        otp = [random.randint(0, 255) for x in range(32)]
        cfg.send(
            self.schema.Struct14590566522150786282.new_message(encrypt=True,
                                                               otp=otp))
        cfgWritten = True

      result = self.readMsg(ep, self.schema.Struct12811887160382076992)
      expectedResults = [x ^ y for (x, y) in zip(otp, blob)]
      print(f"Got {blob}")
      print(f"Exp {expectedResults}")
      assert (list(result.blob) == expectedResults)
