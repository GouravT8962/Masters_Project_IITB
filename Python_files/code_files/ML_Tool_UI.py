# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ML_Tool_UI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(779, 568)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 150, 101, 19))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(70, 200, 121, 19))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(70, 250, 121, 19))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(70, 300, 81, 19))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(70, 350, 121, 19))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(70, 400, 211, 21))
        self.label_6.setObjectName("label_6")
        self.Calculate = QtWidgets.QPushButton(self.centralwidget)
        self.Calculate.setEnabled(True)
        self.Calculate.setGeometry(QtCore.QRect(350, 450, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Calculate.setFont(font)
        self.Calculate.setObjectName("Calculate")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(470, 210, 91, 19))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(470, 130, 91, 19))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(470, 170, 91, 19))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(470, 290, 91, 19))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(470, 250, 91, 19))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(470, 410, 121, 19))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(470, 450, 91, 19))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(470, 330, 131, 19))
        self.label_14.setObjectName("label_14")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(470, 370, 121, 19))
        self.label_16.setObjectName("label_16")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(140, 20, 481, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setAutoFillBackground(False)
        self.label_15.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid grey")
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.op_textbox_1 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_1.setGeometry(QtCore.QRect(600, 120, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_1.setFont(font)
        self.op_textbox_1.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_1.setText("")
        self.op_textbox_1.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_1.setObjectName("op_textbox_1")
        self.op_textbox_2 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_2.setGeometry(QtCore.QRect(600, 160, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_2.setFont(font)
        self.op_textbox_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_2.setText("")
        self.op_textbox_2.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_2.setObjectName("op_textbox_2")
        self.op_textbox_3 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_3.setGeometry(QtCore.QRect(600, 200, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_3.setFont(font)
        self.op_textbox_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_3.setText("")
        self.op_textbox_3.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_3.setObjectName("op_textbox_3")
        self.op_textbox_6 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_6.setGeometry(QtCore.QRect(600, 320, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_6.setFont(font)
        self.op_textbox_6.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_6.setText("")
        self.op_textbox_6.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_6.setObjectName("op_textbox_6")
        self.op_textbox_4 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_4.setGeometry(QtCore.QRect(600, 240, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_4.setFont(font)
        self.op_textbox_4.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_4.setText("")
        self.op_textbox_4.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_4.setObjectName("op_textbox_4")
        self.op_textbox_5 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_5.setGeometry(QtCore.QRect(600, 280, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_5.setFont(font)
        self.op_textbox_5.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_5.setText("")
        self.op_textbox_5.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_5.setObjectName("op_textbox_5")
        self.op_textbox_9 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_9.setGeometry(QtCore.QRect(600, 440, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_9.setFont(font)
        self.op_textbox_9.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_9.setText("")
        self.op_textbox_9.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_9.setObjectName("op_textbox_9")
        self.op_textbox_7 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_7.setGeometry(QtCore.QRect(600, 360, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_7.setFont(font)
        self.op_textbox_7.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_7.setText("")
        self.op_textbox_7.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_7.setObjectName("op_textbox_7")
        self.op_textbox_8 = QtWidgets.QLabel(self.centralwidget)
        self.op_textbox_8.setGeometry(QtCore.QRect(600, 400, 121, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.op_textbox_8.setFont(font)
        self.op_textbox_8.setStyleSheet("background-color: rgb(255, 255, 255);\n"
" border: 1px solid grey;\n"
"font: 10pt \"MS Shell Dlg 2\";\n"
"color: rgb(0, 85, 127);")
        self.op_textbox_8.setText("")
        self.op_textbox_8.setAlignment(QtCore.Qt.AlignCenter)
        self.op_textbox_8.setObjectName("op_textbox_8")
        self.Clear = QtWidgets.QPushButton(self.centralwidget)
        self.Clear.setEnabled(True)
        self.Clear.setGeometry(QtCore.QRect(80, 450, 71, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Clear.setFont(font)
        self.Clear.setObjectName("Clear")
        self.label_input = QtWidgets.QLabel(self.centralwidget)
        self.label_input.setGeometry(QtCore.QRect(130, 70, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_input.setFont(font)
        self.label_input.setAutoFillBackground(False)
        self.label_input.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid grey")
        self.label_input.setAlignment(QtCore.Qt.AlignCenter)
        self.label_input.setObjectName("label_input")
        self.label_output = QtWidgets.QLabel(self.centralwidget)
        self.label_output.setGeometry(QtCore.QRect(470, 70, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_output.setFont(font)
        self.label_output.setAutoFillBackground(False)
        self.label_output.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid grey")
        self.label_output.setAlignment(QtCore.Qt.AlignCenter)
        self.label_output.setObjectName("label_output")
        self.lineEdit_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_1.setGeometry(QtCore.QRect(280, 140, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_1.setFont(font)
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(280, 190, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(280, 290, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(280, 240, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setGeometry(QtCore.QRect(280, 390, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setGeometry(QtCore.QRect(280, 340, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.Set_Default = QtWidgets.QPushButton(self.centralwidget)
        self.Set_Default.setEnabled(True)
        self.Set_Default.setGeometry(QtCore.QRect(190, 450, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.Set_Default.setFont(font)
        self.Set_Default.setObjectName("Set_Default")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 779, 31))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "DC Gain (dB)"))
        self.label_2.setText(_translate("MainWindow", "DC Power (mW)"))
        self.label_3.setText(_translate("MainWindow", "Slew Rate (V/us)"))
        self.label_4.setText(_translate("MainWindow", "UGB (MHz)"))
        self.label_5.setText(_translate("MainWindow", "Phase Margin (°)"))
        self.label_6.setText(_translate("MainWindow", "Noise at 1 Mhz (nV/sqrt(Hz))"))
        self.Calculate.setText(_translate("MainWindow", "Calculate"))
        self.label_7.setText(_translate("MainWindow", "nf_Mn_tail2"))
        self.label_8.setText(_translate("MainWindow", "nf_Mn_drive"))
        self.label_9.setText(_translate("MainWindow", "nf_Mn_tail1"))
        self.label_10.setText(_translate("MainWindow", "nf_ibias"))
        self.label_11.setText(_translate("MainWindow", "nf_Mp_stg2"))
        self.label_12.setText(_translate("MainWindow", "L_Mp_stg1 (nm)"))
        self.label_13.setText(_translate("MainWindow", "mul_cap"))
        self.label_14.setText(_translate("MainWindow", "L_Mn_drive (nm)"))
        self.label_16.setText(_translate("MainWindow", "L_Mn_tail1 (um)"))
        self.label_15.setText(_translate("MainWindow", "Op-Amp Charactestics Prediction Tool"))
        self.Clear.setText(_translate("MainWindow", "Clear"))
        self.label_input.setText(_translate("MainWindow", "Input Variables"))
        self.label_output.setText(_translate("MainWindow", "Output Variables"))
        self.lineEdit_1.setText(_translate("MainWindow", "48"))
        self.lineEdit_2.setText(_translate("MainWindow", "1.5"))
        self.lineEdit_4.setText(_translate("MainWindow", "250"))
        self.lineEdit_3.setText(_translate("MainWindow", "200"))
        self.lineEdit_6.setText(_translate("MainWindow", "10"))
        self.lineEdit_5.setText(_translate("MainWindow", "65"))
        self.Set_Default.setText(_translate("MainWindow", "Set Default"))

