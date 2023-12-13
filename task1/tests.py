import unittest
from solution import *


class Tests(unittest.TestCase):
    def testColor(self):
        v = fr"^({COLOR_REGEXP})$"
        self.assertRegex("#21f48D", v)
        self.assertRegex("#888", v)
        self.assertRegex("rgb(255, 255,255)", v)
        self.assertRegex("rgb(10%, 20%, 0%)", v)
        self.assertRegex("hsl(200,100%,50%)", v)
        self.assertRegex("hsl(0, 0%, 0%)", v)

        self.assertNotRegex("#2345", v)
        self.assertNotRegex("ffffff", v)
        self.assertNotRegex("rgb(257, 50, 10)", v)
        self.assertNotRegex("rgb(0, 50, 130%)", v)
        self.assertNotRegex("hsl(20, 10, 0.5)", v)
        self.assertNotRegex("hsl(34%, 20%, 50%)", v)

    def testPassword(self):
        v = fr'^({PASSWORD_REGEXP})$'
        self.assertRegex("rtG3FG!Tr^e", v)
        self.assertRegex("aA1!*!1Aa", v)
        self.assertRegex("oF^a1D@y5e6", v)
        self.assertRegex("enroi#$rkdeR#$092uwedchf34tguv394h", v)
        self.assertRegex("pAs$w0rd!", v)

        self.assertNotRegex("пароль", v)
        self.assertNotRegex("password", v)
        self.assertNotRegex("qwerty", v)
        self.assertNotRegex("lOngPa$$W0Rd", v)
        self.assertNotRegex("pAs$w0rd", v)
        self.assertNotRegex("pAs^w0rd", v)
        self.assertNotRegex("pAs*w0rd", v)

    def testDate(self):
        v = fr'^({DATES_REGEXP})$'
        self.assertRegex("20 января 1806", v)
        self.assertRegex("14 сентября 2023", v)
        self.assertRegex("5 февраля 1995", v)
        self.assertRegex("01 апреля 2012", v)
        self.assertRegex("September 14, 2023", v)
        self.assertRegex("February 5, 1995", v)
        self.assertRegex("April 01, 2012", v)
        self.assertRegex("Sep 14, 2023", v)
        self.assertRegex("Feb 5, 1995", v)
        self.assertRegex("Apr 01, 2012", v)
        self.assertRegex("1924, July 25", v)
        self.assertRegex("26/09/1635", v)
        self.assertRegex("3.1.1506", v)
        self.assertRegex("2023.09.14", v)
        self.assertRegex("1995-02-5", v)

        self.assertNotRegex("25.08-1002", v)
        self.assertNotRegex("декабря 19, 1838", v)
        self.assertNotRegex("8.20.1973", v)
        self.assertNotRegex("Jun 7, -1563", v)


unittest.main()
